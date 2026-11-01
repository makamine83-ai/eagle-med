// hospital_channel.cc
// ns-3.41 packet-level calibration of the EAGLE-Med link model.
// Topology: W wards, each a non-overlapping 802.11ac BSS (one AP, G edge gateways,
// B background stations); APs and S edge servers share a 1 Gb/s wired backbone.
// Small UDP probes measure one-way delay and loss per link class and access category
// while background traffic loads each BSS. Results are written as quantiles.
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>
using namespace ns3;

static const uint16_t PROBE_PORT = 9000;
static const uint16_t BG_PORT = 9001;

struct Key
{
    std::string cls;
    int ac;
    bool operator<(const Key& o) const { return cls != o.cls ? cls < o.cls : ac < o.ac; }
};
struct Stat
{
    uint64_t sent = 0, recv = 0;
    std::vector<double> d;
};
static std::map<Key, Stat> g_stat;

// header carried in the probe payload
struct ProbeHdr
{
    double t;
    int32_t cls;
    int32_t ac;
};
static std::vector<std::string> g_classes = {"WW_same", "WW_cross", "WS", "SW", "SS"};
static std::map<std::string, uint64_t> g_acCount;

static void
CountAc(std::string ac, Ptr<const WifiMpdu> mpdu)
{
    g_acCount[ac]++;
}

static void
RxProbe(Ptr<Socket> s)
{
    Ptr<Packet> p;
    Address from;
    while ((p = s->RecvFrom(from)))
    {
        ProbeHdr h;
        p->CopyData((uint8_t*)&h, sizeof(h));
        auto& st = g_stat[{g_classes[h.cls], h.ac}];
        st.recv++;
        st.d.push_back(Simulator::Now().GetSeconds() - h.t);
    }
}

class Prober : public Object
{
  public:
    Ptr<Node> node;
    std::vector<std::pair<Ipv4Address, int>> peers; // peer address, link class
    double rate = 10;
    uint32_t size = 200;
    double stop = 0;
    Ptr<ExponentialRandomVariable> iat;
    Ptr<UniformRandomVariable> u;
    std::map<std::pair<int, int>, Ptr<Socket>> sock; // (peer index, access category)

    // In ns-3.41 SendTo takes the TOS from the address, and Connect overwrites the socket TOS with it,
    // so the mark is set after Connect. The Wi-Fi queue selection derives the access category from it.
    Ptr<Socket> Sock(int idx, int a)
    {
        auto it = sock.find({idx, a});
        if (it != sock.end())
        {
            return it->second;
        }
        Ptr<Socket> s = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
        s->Bind();
        s->Connect(InetSocketAddress(peers[idx].first, PROBE_PORT));
        s->SetIpTos(a == 0 ? 0x00 : 0xc0); // AC_BE vs AC_VO; must follow Connect, which resets the TOS
        sock[{idx, a}] = s;
        return s;
    }

    void Start()
    {
        iat = CreateObject<ExponentialRandomVariable>();
        iat->SetAttribute("Mean", DoubleValue(1.0 / rate));
        u = CreateObject<UniformRandomVariable>();
        Simulator::Schedule(Seconds(iat->GetValue()), &Prober::Fire, this);
    }

    void Fire()
    {
        if (Simulator::Now().GetSeconds() > stop)
        {
            return;
        }
        int idx = std::min<int>((int)(u->GetValue() * peers.size()), peers.size() - 1);
        int a = u->GetValue() < 0.5 ? 0 : 1;
        ProbeHdr h{Simulator::Now().GetSeconds(), peers[idx].second, a};
        std::vector<uint8_t> buf(size, 0);
        std::memcpy(buf.data(), &h, sizeof(h));
        Ptr<Packet> p = Create<Packet>(buf.data(), size);
        Sock(idx, a)->Send(p);
        g_stat[{g_classes[peers[idx].second], a}].sent++;
        Simulator::Schedule(Seconds(iat->GetValue()), &Prober::Fire, this);
    }
};

int
main(int argc, char** argv)
{
    uint32_t wards = 4, gws = 3, bg = 4, servers = 2, seed = 1, probeBytes = 200;
    double loadMbps = 10, simTime = 60, warm = 5, probeRate = 12, radius = 18;
    std::string out = "cal.csv";
    CommandLine cmd;
    cmd.AddValue("wards", "wards", wards);
    cmd.AddValue("gws", "gateways per ward", gws);
    cmd.AddValue("bg", "background stations per ward", bg);
    cmd.AddValue("servers", "edge servers", servers);
    cmd.AddValue("seed", "run", seed);
    cmd.AddValue("load", "background Mb/s per ward", loadMbps);
    cmd.AddValue("time", "sim time", simTime);
    cmd.AddValue("probeRate", "probes/s per node", probeRate);
    cmd.AddValue("probeBytes", "probe size", probeBytes);
    cmd.AddValue("radius", "ward radius m", radius);
    cmd.AddValue("out", "output", out);
    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(20261031);
    RngSeedManager::SetRun(seed);

    NodeContainer srv;
    srv.Create(servers);
    NodeContainer aps;
    aps.Create(wards);
    std::vector<NodeContainer> gw(wards), bgs(wards);
    InternetStackHelper inet;
    inet.Install(srv);
    inet.Install(aps);
    Ipv4AddressHelper addr;

    // wired backbone
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(20)));
    NodeContainer lan;
    lan.Add(srv);
    lan.Add(aps);
    NetDeviceContainer lanDev = csma.Install(lan);
    addr.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer lanIf = addr.Assign(lanDev);

    std::vector<std::vector<Ipv4Address>> gwAddr(wards);
    MobilityHelper fixed;
    fixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<UniformRandomVariable> pos = CreateObject<UniformRandomVariable>();
    for (uint32_t w = 0; w < wards; w++)
    {
        gw[w].Create(gws);
        bgs[w].Create(bg);
        inet.Install(gw[w]);
        inet.Install(bgs[w]);
        YansWifiChannelHelper ch;
        ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
        ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                              "Exponent", DoubleValue(3.0),
                              "ReferenceLoss", DoubleValue(46.7));
        ch.AddPropagationLoss("ns3::NakagamiPropagationLossModel",
                              "m0", DoubleValue(1.5),
                              "m1", DoubleValue(1.5),
                              "m2", DoubleValue(1.5));
        YansWifiPhyHelper phy;
        phy.SetChannel(ch.Create());
        phy.Set("ChannelSettings", StringValue("{0, 40, BAND_5GHZ, 0}"));
        phy.Set("TxPowerStart", DoubleValue(18));
        phy.Set("TxPowerEnd", DoubleValue(18));
        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211ac);
        wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
        WifiMacHelper mac;
        Ssid ssid("ward" + std::to_string(w));
        mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
        NodeContainer sta;
        sta.Add(gw[w]);
        sta.Add(bgs[w]);
        NetDeviceContainer staDev = wifi.Install(phy, mac, sta);
        mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
        NetDeviceContainer apDev = wifi.Install(phy, mac, NodeContainer(aps.Get(w)));
        fixed.Install(aps.Get(w));
        aps.Get(w)->GetObject<MobilityModel>()->SetPosition(Vector(w * 60.0, 0, 3));
        fixed.Install(sta);
        for (uint32_t i = 0; i < sta.GetN(); i++)
        {
            double r = radius * std::sqrt(pos->GetValue(0.05, 1.0));
            double a = pos->GetValue(0, 6.28318);
            sta.Get(i)->GetObject<MobilityModel>()->SetPosition(
                Vector(w * 60.0 + r * std::cos(a), r * std::sin(a), 1));
        }
        std::string base = "10.1." + std::to_string(w + 1) + ".0";
        addr.SetBase(base.c_str(), "255.255.255.0");
        addr.Assign(apDev);
        Ipv4InterfaceContainer staIf = addr.Assign(staDev);
        for (uint32_t i = 0; i < gws; i++)
        {
            gwAddr[w].push_back(staIf.GetAddress(i));
        }
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // background: monitoring/imaging streams from each ward towards a server, best effort
    Ipv4Address sinkAddr = lanIf.GetAddress(0);
    PacketSinkHelper bgSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), BG_PORT));
    bgSink.Install(srv.Get(0)).Start(Seconds(0.5));
    if (loadMbps > 0)
    {
        double perSta = loadMbps / bg;
        for (uint32_t w = 0; w < wards; w++)
        {
            OnOffHelper on("ns3::UdpSocketFactory", InetSocketAddress(sinkAddr, BG_PORT));
            on.SetAttribute("PacketSize", UintegerValue(1200));
            on.SetAttribute("DataRate", DataRateValue(DataRate((uint64_t)(perSta * 2e6))));
            on.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            on.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            ApplicationContainer a = on.Install(bgs[w]);
            a.Start(Seconds(1.0));
            a.Stop(Seconds(simTime));
        }
    }

    // probe sinks
    NodeContainer all;
    all.Add(srv);
    for (uint32_t w = 0; w < wards; w++)
    {
        all.Add(gw[w]);
    }
    for (uint32_t i = 0; i < all.GetN(); i++)
    {
        Ptr<Socket> s = Socket::CreateSocket(all.Get(i), UdpSocketFactory::GetTypeId());
        s->Bind(InetSocketAddress(Ipv4Address::GetAny(), PROBE_PORT));
        s->SetRecvCallback(MakeCallback(&RxProbe));
    }
    // probers
    std::vector<Ptr<Prober>> pr;
    for (uint32_t w = 0; w < wards; w++)
    {
        for (uint32_t i = 0; i < gws; i++)
        {
            Ptr<Prober> p = CreateObject<Prober>();
            p->node = gw[w].Get(i);
            p->rate = probeRate;
            p->size = probeBytes;
            p->stop = simTime - 1;
            for (uint32_t k = 0; k < gws; k++)
            {
                if (k != i)
                {
                    p->peers.push_back({gwAddr[w][k], 0});
                }
            }
            for (uint32_t x = 0; x < wards; x++)
            {
                if (x != w)
                {
                    p->peers.push_back({gwAddr[x][i], 1});
                }
            }
            for (uint32_t s = 0; s < servers; s++)
            {
                p->peers.push_back({lanIf.GetAddress(s), 2});
            }
            pr.push_back(p);
            Simulator::Schedule(Seconds(warm), &Prober::Start, p);
        }
    }
    for (uint32_t s = 0; s < servers; s++)
    {
        Ptr<Prober> p = CreateObject<Prober>();
        p->node = srv.Get(s);
        p->rate = probeRate;
        p->size = probeBytes;
        p->stop = simTime - 1;
        for (uint32_t w = 0; w < wards; w++)
        {
            for (uint32_t i = 0; i < gws; i++)
            {
                p->peers.push_back({gwAddr[w][i], 3});
            }
        }
        for (uint32_t k = 0; k < servers; k++)
        {
            if (k != s)
            {
                p->peers.push_back({lanIf.GetAddress(k), 4});
            }
        }
        pr.push_back(p);
        Simulator::Schedule(Seconds(warm), &Prober::Start, p);
    }
    for (std::string ac : {"BE", "VO"})
    {
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/" + ac + "_Txop/Queue/Enqueue",
                                      MakeBoundCallback(&CountAc, ac));
    }
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    for (auto& kv : g_acCount)
    {
        std::cout << "enqueued " << kv.first << " " << kv.second << std::endl;
    }
    std::ofstream f(out);
    f << "cls,ac,load,seed,sent,recv";
    for (int q = 0; q <= 100; q++)
    {
        f << ",q" << q;
    }
    f << "\n";
    for (auto& kv : g_stat)
    {
        auto& st = kv.second;
        std::sort(st.d.begin(), st.d.end());
        f << kv.first.cls << "," << kv.first.ac << "," << loadMbps << "," << seed << "," << st.sent << ","
          << st.recv;
        for (int q = 0; q <= 100; q++)
        {
            double v = st.d.empty() ? -1 : st.d[std::min<size_t>(st.d.size() - 1, q * st.d.size() / 100)];
            f << "," << v;
        }
        f << "\n";
    }
    Simulator::Destroy();
    return 0;
}
