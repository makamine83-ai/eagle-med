// closedloop.cc
// Closed-loop ns-3.41 cross-validation of the Python simulator.
// Same hospital topology as hospital_channel.cc, with the election and alert logic running as
// real UDP traffic on the simulated Wi-Fi and wired network. Trust and scoring are replaced by a
// fixed candidate rank, because this program only checks timing: failover and alert latency.
//   mode 0: Raft timers, alerts relayed through the leader
//   mode 1: EAGLE-Med timers (urgent heartbeat, rank-ordered campaign), leader-independent alerts
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
#include <set>
#include <vector>
using namespace ns3;

static const uint16_t CTL_PORT = 7000, ALERT_PORT = 7001, BG_PORT = 9001;
enum
{
    HB,
    ACK,
    PV,
    PVR,
    RV,
    RVR,
    ALERT,
    AACK
};
struct Msg
{
    int32_t type, src, term, arg;
    double t;
};
static const uint32_t SZ[] = {96, 72, 96, 72, 96, 72, 192, 72};

static int g_mode = 1;
static double g_crash = 20, g_failover = -1, g_crashed = -1;
static uint32_t g_n = 0, g_q = 0;
static std::vector<Ipv4Address> g_addr, g_sink;
static std::map<int, double> g_alertBirth, g_alertFirst;
static std::set<int> g_alertAcked;
class Member;
static std::vector<Ptr<Member>> g_m;

class Member : public Object
{
  public:
    int id = 0, rank = 0, role = 0, term = 0, voted = -1, leader = -1;
    bool up = true, serving = false;
    Ptr<Node> node;
    Ptr<Socket> ctl, alr;
    std::map<uint32_t, Ptr<Socket>> alrTx; // connected AC_VO sockets
    double lastHb = 0, toCur = 1, hbRx = -1e9; // election timer, and last real heartbeat for the pre-vote check
    int pvTerm = -1;
    std::set<int> pv, votes, acks;
    Ptr<UniformRandomVariable> u;
    uint32_t tick = 0;

    double Thb() { return g_mode ? 0.10 : 0.20; }
    double Tmin() { return 3 * Thb(); }
    double Timeout()
    {
        if (g_mode)
        {
            return Tmin() + 0.10 * rank + 0.05 * u->GetValue();
        }
        return Tmin() * (1 + u->GetValue());
    }

    void Init()
    {
        u = CreateObject<UniformRandomVariable>();
        ctl = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
        ctl->Bind(InetSocketAddress(Ipv4Address::GetAny(), CTL_PORT));
        ctl->SetRecvCallback(MakeCallback(&Member::Rx, this));
        ctl->SetIpTos(0x00);
        alr = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
        alr->Bind(InetSocketAddress(Ipv4Address::GetAny(), ALERT_PORT));
        alr->SetRecvCallback(MakeCallback(&Member::Rx, this));
        lastHb = Simulator::Now().GetSeconds();
        toCur = Timeout();
        Simulator::Schedule(Seconds(toCur), &Member::Timer, this);
    }

    void Send(Ipv4Address a, int type, int termv, int arg, double t = 0)
    {
        if (!up)
        {
            return;
        }
        Msg m{type, id, termv, arg, t};
        std::vector<uint8_t> buf(SZ[type], 0);
        std::memcpy(buf.data(), &m, sizeof(m));
        if (type == ALERT || type == AACK)
        {
            uint32_t key = a.Get();
            if (!alrTx.count(key))
            {
                Ptr<Socket> s = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
                s->Bind();
                s->Connect(InetSocketAddress(a, ALERT_PORT));
                s->SetIpTos(0xc0); // must follow Connect, which resets the TOS in ns-3.41
                alrTx[key] = s;
            }
            alrTx[key]->Send(Create<Packet>(buf.data(), SZ[type]));
            return;
        }
        ctl->SendTo(Create<Packet>(buf.data(), SZ[type]), 0, InetSocketAddress(a, CTL_PORT));
    }

    void Bcast(int type, int termv, int arg, double t = 0)
    {
        for (uint32_t i = 0; i < g_n; i++)
        {
            if ((int)i != id)
            {
                Send(g_addr[i], type, termv, arg, t);
            }
        }
    }

    void Timer()
    {
        if (!up)
        {
            return;
        }
        double now = Simulator::Now().GetSeconds();
        if (role != 2 && now + 1e-9 >= lastHb + toCur)
        {
            pvTerm = term + 1;
            pv = {id};
            Bcast(PV, pvTerm, 0);
            lastHb = now;
            toCur = Timeout();
        }
        double due = std::max(lastHb + toCur, now + 0.001);
        Simulator::Schedule(Seconds(due - now), &Member::Timer, this);
    }

    void Tick(uint32_t gen)
    {
        if (gen != tick || role != 2 || !up)
        {
            return;
        }
        Bcast(HB, term, 0, Simulator::Now().GetSeconds());
        Simulator::Schedule(Seconds(Thb()), &Member::Tick, this, gen);
    }

    void StepDown(int t)
    {
        role = 0;
        term = t;
        voted = -1;
        serving = false;
        tick++;
    }

    void Rx(Ptr<Socket> s)
    {
        Ptr<Packet> p;
        Address from;
        while ((p = s->RecvFrom(from)))
        {
            if (!up)
            {
                continue;
            }
            Msg m;
            p->CopyData((uint8_t*)&m, sizeof(m));
            Handle(m);
        }
    }

    void Handle(const Msg& m)
    {
        double now = Simulator::Now().GetSeconds();
        switch (m.type)
        {
        case HB:
            if (m.term < term)
            {
                break;
            }
            if (m.term > term || role != 0)
            {
                StepDown(m.term);
            }
            leader = m.src;
            lastHb = hbRx = now;
            Send(g_addr[m.src], ACK, m.term, 0);
            break;
        case ACK:
            if (role == 2 && m.term == term && !serving)
            {
                acks.insert(m.src);
                if (acks.size() + 1 >= g_q)
                {
                    serving = true;
                    if (now > g_crash && g_failover < 0)
                    {
                        g_failover = now - g_crash;
                    }
                }
            }
            break;
        case PV: {
            bool recent = (now - hbRx < Tmin()) && leader >= 0;
            Send(g_addr[m.src], PVR, m.term, (m.term > term && !recent) ? 1 : 0);
            break;
        }
        case PVR:
            if (m.term == pvTerm && m.arg && role != 2)
            {
                pv.insert(m.src);
                if (pv.size() >= g_q && term < m.term)
                {
                    term = m.term;
                    role = 1;
                    voted = id;
                    votes = {id};
                    Bcast(RV, term, 0);
                }
            }
            break;
        case RV: {
            if (m.term > term && role != 2 && (now - hbRx < Tmin()) && leader >= 0)
            {
                Send(g_addr[m.src], RVR, term, 0); // the pre-vote rule holds at the vote too
                break;
            }
            if (m.term > term)
            {
                StepDown(m.term);
            }
            bool g = m.term == term && (voted == -1 || voted == m.src);
            if (g)
            {
                voted = m.src;
                lastHb = now;
            }
            Send(g_addr[m.src], RVR, term, g ? 1 : 0);
            break;
        }
        case RVR:
            if (m.term > term)
            {
                StepDown(m.term);
            }
            else if (role == 1 && m.term == term && m.arg)
            {
                votes.insert(m.src);
                if (votes.size() >= g_q)
                {
                    role = 2;
                    leader = id;
                    acks.clear();
                    tick++;
                    Tick(tick);
                }
            }
            break;
        case ALERT: // m.arg: alert id; m.term: 1 asks a serving leader or relay to forward
            if (m.term == 1 && (serving || g_mode == 1))
            {
                for (auto& k : g_sink)
                {
                    Msg f = m;
                    Send(k, ALERT, 0, m.arg, m.t);
                }
            }
            break;
        default:
            break;
        }
    }
};

class Sink : public Object
{
  public:
    Ptr<Socket> s;
    void Init(Ptr<Node> n)
    {
        s = Socket::CreateSocket(n, UdpSocketFactory::GetTypeId());
        s->Bind(InetSocketAddress(Ipv4Address::GetAny(), ALERT_PORT));
        s->SetRecvCallback(MakeCallback(&Sink::Rx, this));
    }
    void Rx(Ptr<Socket> so)
    {
        Ptr<Packet> p;
        Address from;
        while ((p = so->RecvFrom(from)))
        {
            Msg m;
            p->CopyData((uint8_t*)&m, sizeof(m));
            if (!g_alertFirst.count(m.arg))
            {
                g_alertFirst[m.arg] = Simulator::Now().GetSeconds();
                g_alertAcked.insert(m.arg); // sinks acknowledge; one-way ack delay is negligible here
            }
        }
    }
};

static void
AlertTry(int aid, int src, Ptr<UniformRandomVariable> u)
{
    if (g_alertAcked.count(aid) || Simulator::Now().GetSeconds() - g_alertBirth[aid] > 10)
    {
        return;
    }
    Ptr<Member> m = g_m[src];
    if (m->up)
    {
        if (g_mode == 1)
        {
            for (auto& k : g_sink)
            {
                m->Send(k, ALERT, 0, aid, g_alertBirth[aid]);
            }
            int rly = 2 + (int)(u->GetValue() * (g_n - 2));
            m->Send(g_addr[rly], ALERT, 1, aid, g_alertBirth[aid]);
        }
        else if (m->leader >= 0)
        {
            m->Send(g_addr[m->leader], ALERT, 1, aid, g_alertBirth[aid]);
        }
    }
    Simulator::Schedule(Seconds(0.2), &AlertTry, aid, src, u);
}

static int g_aid = 0;

static void
AlertGen(double stop, std::vector<int> src, Ptr<ExponentialRandomVariable> iat, Ptr<UniformRandomVariable> u)
{
    double now = Simulator::Now().GetSeconds();
    if (now > stop)
    {
        return;
    }
    int aid = ++g_aid;
    g_alertBirth[aid] = now;
    AlertTry(aid, src[(int)(u->GetValue() * src.size())], u);
    Simulator::Schedule(Seconds(iat->GetValue()), &AlertGen, stop, src, iat, u);
}

static void
CrashLeader()
{
    for (auto& m : g_m)
    {
        if (m->serving)
        {
            m->up = false;
            m->serving = false;
            g_crashed = Simulator::Now().GetSeconds();
            return;
        }
    }
}

int
main(int argc, char** argv)
{
    uint32_t wards = 4, gwsPerWard = 0, bg = 4, servers = 2, seed = 1, n = 20;
    double loadMbps = 20, simTime = 40, radius = 18;
    std::string out = "cl.csv";
    CommandLine cmd;
    cmd.AddValue("mode", "0 raft, 1 eagle", g_mode);
    cmd.AddValue("seed", "run", seed);
    cmd.AddValue("load", "background Mb/s per ward", loadMbps);
    cmd.AddValue("n", "members", n);
    cmd.AddValue("time", "sim time", simTime);
    cmd.AddValue("crash", "crash time", g_crash);
    cmd.AddValue("out", "output", out);
    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(20261101);
    RngSeedManager::SetRun(seed);
    g_n = n;
    g_q = n / 2 + 1;
    gwsPerWard = (n - servers) / wards;
    uint32_t extra = (n - servers) % wards; // the first wards take the remainder, so that the group has exactly n members

    NodeContainer srv, sinks, aps;
    srv.Create(servers);
    sinks.Create(2);
    aps.Create(wards);
    std::vector<NodeContainer> gw(wards), bgs(wards);
    InternetStackHelper inet;
    inet.Install(srv);
    inet.Install(sinks);
    inet.Install(aps);
    Ipv4AddressHelper addr;
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MicroSeconds(20)));
    NodeContainer lan;
    lan.Add(srv);
    lan.Add(sinks);
    lan.Add(aps);
    NetDeviceContainer lanDev = csma.Install(lan);
    addr.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer lanIf = addr.Assign(lanDev);
    for (uint32_t s = 0; s < servers; s++)
    {
        g_addr.push_back(lanIf.GetAddress(s));
    }
    g_sink = {lanIf.GetAddress(servers), lanIf.GetAddress(servers + 1)};

    MobilityHelper fixed;
    fixed.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<UniformRandomVariable> pos = CreateObject<UniformRandomVariable>();
    NodeContainer members;
    members.Add(srv);
    std::vector<int> ward0;
    for (uint32_t w = 0; w < wards; w++)
    {
        uint32_t gws = gwsPerWard + (w < extra ? 1 : 0);
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
            if (w == 0)
            {
                ward0.push_back(g_addr.size());
            }
            g_addr.push_back(staIf.GetAddress(i));
            members.Add(gw[w].Get(i));
        }
    }
    g_n = g_addr.size();
    g_q = g_n / 2 + 1;
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    PacketSinkHelper bgSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), BG_PORT));
    bgSink.Install(srv.Get(0)).Start(Seconds(0.5));
    double perSta = loadMbps / bg;
    if (loadMbps > 0)
    {
        for (uint32_t w = 0; w < wards; w++)
        {
            OnOffHelper on("ns3::UdpSocketFactory", InetSocketAddress(lanIf.GetAddress(0), BG_PORT));
            on.SetAttribute("PacketSize", UintegerValue(1200));
            on.SetAttribute("DataRate", DataRateValue(DataRate((uint64_t)(perSta * 2e6))));
            on.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            on.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            ApplicationContainer a = on.Install(bgs[w]);
            a.Start(Seconds(1.0));
            a.Stop(Seconds(simTime));
        }
    }

    for (uint32_t i = 0; i < g_n; i++)
    {
        Ptr<Member> m = CreateObject<Member>();
        m->id = i;
        m->rank = i; // fixed rank stands in for the score ordering: servers first
        m->node = members.Get(i);
        g_m.push_back(m);
        Simulator::Schedule(Seconds(2.0), &Member::Init, m);
    }
    std::vector<Ptr<Sink>> sk;
    for (uint32_t i = 0; i < 2; i++)
    {
        Ptr<Sink> s = CreateObject<Sink>();
        s->Init(sinks.Get(i));
        sk.push_back(s);
    }
    Simulator::Schedule(Seconds(g_crash), &CrashLeader);
    Ptr<ExponentialRandomVariable> iat = CreateObject<ExponentialRandomVariable>();
    iat->SetAttribute("Mean", DoubleValue(0.5));
    Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable>();
    Simulator::Schedule(Seconds(g_crash - 8), &AlertGen, g_crash + 12, ward0, iat, u);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    std::ofstream f(out);
    f << "mode,seed,failover_s,alert_id,birth,latency,crash_s,members\n";
    for (auto& kv : g_alertBirth)
    {
        double lat = g_alertFirst.count(kv.first) ? g_alertFirst[kv.first] - kv.second : -1;
        f << g_mode << "," << seed << "," << g_failover << "," << kv.first << "," << kv.second << "," << lat << "," << g_crashed << "," << g_n << "\n";
    }
    Simulator::Destroy();
    return 0;
}
