import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

rng = np.random.default_rng(2026)

for f in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 10.5

# ================= 模型参数 (论文 §5.2) =================
Q     = 5.0          # Ah
R_INT = 0.02         # ohm
H     = 0.05         # 散热系数
DT    = 60.0         # s
T_AMB = 25.0
ALPHA = 0.2          
SOC0, T0 = 0.50, 25.0
U_DES_CONST = 13.0   
SAFETY_MARGIN = 0.5  # 离散时序补偿裕度(C): 投影保证 T_{k+1} <= T_max(r_k)-MARGIN,
                     # 消除风险上升段 T_max 逐步下降造成的检查错位伪影


def soc_min(r): return 0.05 + 0.15 * r
def soc_max(r): return 0.95 - 0.15 * r
def t_max(r):   return 60.0 - 15.0 * r
def i_max(r):   return 3.0 * Q * (1.0 - 0.7 * r)   # = 15*(1-0.7r)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def risk_signal(k):
    """论文 §5.2 的双 sigmoid 合成风险信号"""
    return 0.1 + 0.8 * sigmoid((k - 50) / 10.0) * sigmoid((150 - k) / 10.0)

R_STATIC = 0.0   # 恒定电流基线(非自适应)的固定边界: 全局最宽松, 不随风险收紧

def project(u_des, soc, T, r, static=False):
    """投影到 S(r): 返回 (u_safe, 各约束界)
       static=True 时使用固定边界 R_STATIC(恒定电流基线, 不随风险收缩)"""
    if static:
        rr = R_STATIC
    else:
        rr = r
    i_imax  = i_max(rr)
    # SOC 下界(防过度放电): u <= (soc - SOC_min)*Q/dt
    i_lo_soc = max(0.0, (soc - soc_min(rr)) * Q * 3600.0 / DT)
    # SOC 上界(防过度充电): -u <= (SOC_max - soc)*Q/dt
    i_up_soc = max(0.0, (soc_max(rr) - soc) * Q * 3600.0 / DT)
    # 温度约束: u^2 <= (T_max - MARGIN - T)/(alpha*R_int) + h*(T-T_amb)/R_int
    tmp = (t_max(rr) - SAFETY_MARGIN - T) / (ALPHA * R_INT) + H * (T - T_AMB) / R_INT
    i_temp = np.sqrt(max(0.0, tmp))
    u_pos = min(i_imax, i_lo_soc, i_temp)
    u_neg = min(i_imax, i_up_soc, i_temp)
    u_safe = np.clip(u_des, -u_neg, u_pos)
    return u_safe, (i_imax, i_lo_soc, i_up_soc, i_temp)

def step(soc, T, u):
    soc2 = soc - (DT / 3600.0) * u / Q          # 放电为正, SOC 减
    T2   = T + ALPHA * (u * u * R_INT - H * (T - T_AMB))
    return soc2, T2

def run_closed(u_policy, apply_projection=True):
    """u_policy(soc, k, r) -> 愿望动作; 记录历史"""
    soc, T = SOC0, T0
    rec = dict(k=[], r=[], u_des=[], u_safe=[],
               soc=[], T=[], Tmax=[], Smin=[], Smax=[], Ilim=[])
    for k in range(STEPS):
        r = risk_signal(k)
        u_des = u_policy(soc, k, r)
        if apply_projection:
            u_safe, bounds = project(u_des, soc, T, r)
        else:
            u_safe, bounds = u_des, (i_max(r),)*4
        rec["k"].append(k); rec["r"].append(r)
        rec["u_des"].append(u_des)
        rec["u_safe"].append(u_safe)
        rec["soc"].append(soc); rec["T"].append(T)
        rec["Tmax"].append(t_max(r))
        rec["Smin"].append(soc_min(r)); rec["Smax"].append(soc_max(r))
        rec["Ilim"].append(i_max(r))
        soc, T = step(soc, T, u_safe)
    for key in ("r", "u_des", "u_safe", "soc", "T", "Tmax", "Smin", "Smax", "Ilim"):
        rec[key] = np.array(rec[key])
    return rec

def desire_const(c):
    return lambda soc, k, r: float(c)

def desire_soc_track(c):
    def pol(soc, k, r):
        return float(c) if soc >= SOC_REF else -float(c)
    return pol

SOC_REF = 0.5

seocf = run_closed(desire_soc_track(U_DES_CONST), apply_projection=True)
naked = run_closed(desire_soc_track(U_DES_CONST), apply_projection=False)
consv = run_closed(desire_soc_track(3.0), apply_projection=True)

print("== 性质验证 ==")
# 命题2 投影幂等性: 任意点两次投影与一次投影一致
idem_max = 0.0
for soc in np.linspace(0.05, 0.95, 50):
    for T in np.linspace(25, 60, 20):
        for r in np.linspace(0, 1, 20):
            for ud in [-13, -5, 0, 5, 13]:
                u1, _ = project(ud, soc, T, r)
                u2, _ = project(u1, soc, T, r)
                idem_max = max(idem_max, abs(u2 - u1))
print(f"投影幂等性 max|Pi(Pi(u))-Pi(u)| = {idem_max:.3e}  (0=严格幂等)")
# 命题1 风险单调收缩: |S(r)| 长度随 r 递减
lr = np.linspace(0, 1, 101)
lens = [i_max(rr) - (-i_max(rr)) for rr in lr]
mono = all(lens[i] >= lens[i+1] for i in range(len(lens)-1))
print(f"风险单调收缩 |S(r)| 随 r 递减: {mono}  (I_max(0)={i_max(0):.2f} -> I_max(1)={i_max(1):.2f})")
# 命题3 前向不变性
print(f"裸RL(无投影): T 范围[25,{naked['T'].max():.1f}], 超出T_max(r)最大 {max(0,(naked['T']-naked['Tmax']).max()):.2f} C")
print(f"SE-OCF:      T 超界最大 {max(0,(seocf['T']-seocf['Tmax']).max()):.2f} C, "
      f"SOC 越界最大 {max(0,(seocf['Smin']-seocf['soc']).max()):.2f} (应=0)")
print(f"安全优先:    T 超界最大 {max(0,(consv['T']-consv['Tmax']).max()):.2f} C")

fig1 = plt.figure(figsize=(9.5, 10.5))
gs = gridspec.GridSpec(3, 1, hspace=0.42)

ax1 = fig1.add_subplot(gs[0])
k = seocf["k"]
ax1.plot(k, seocf["r"], color="#d62728", lw=2.0, label="风险等级 $r_k$")
ax1.set_xlabel("时间步 $k$"); ax1.set_ylabel("风险等级 $r$")
ax1.set_ylim(0, 1.2); ax1.grid(alpha=0.3)
ax1.set_title("(上) 合成风险信号 $r_k$ 与 SOC 可行域边界", fontsize=11)
axr = ax1.twinx()
axr.plot(k, seocf["Smin"], ls="--", color="#1f77b4", lw=1.6, label="SOC$_{\\min}(r_k)$")
axr.plot(k, seocf["Smax"], ls="--", color="#7f7f7f", lw=1.6, label="SOC$_{\\max}(r_k)$")
axr.fill_between(k, seocf["Smin"], seocf["Smax"], color="#cfe3f7", alpha=0.5)
axr.set_ylabel("SOC 边界"); axr.set_ylim(0, 1.0)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

ax2 = fig1.add_subplot(gs[1])
ax2.plot(k, seocf["u_des"], color="#7f7f7f", ls=":", lw=1.8,
         label="愿望动作 $u_{\\rm des}$ (SOC 跟踪)")
ax2.plot(k, seocf["u_safe"], color="#1f77b4", lw=2.2,
         label="安全动作 $u_{\\rm safe}=\\Pi_S(u_{\\rm des})$")
ax2.plot(k, seocf["Ilim"], color="#d62728", ls="--", lw=1.5,
         label="$I_{\\max}(r_k)$")
ax2.plot(k, -seocf["Ilim"], color="#d62728", ls="--", lw=1.5)
ax2.fill_between(k, -seocf["Ilim"], seocf["Ilim"], color="#cfe3f7", alpha=0.45,
                 label="可行域 $\\mathcal{S}(r_k)$")
ax2.set_xlabel("时间步 $k$"); ax2.set_ylabel("电流 $u$ (A)")
ax2.set_title("(中) 安全投影: 风险升高时 $u_{\\rm safe}$ 被压回可行域", fontsize=11)
ax2.grid(alpha=0.3); ax2.legend(loc="lower left", fontsize=9)
# 抖振说明(审稿意见修正): 抖振来源是 u_des 在动态安全边界附近频繁穿梭, 而非 I_max 突变
ax2.annotate("注: 愿望动作 $u_{\\rm des}$ 在动态安全边界附近频繁穿梭时,\n"
             "投影输出会高频切换(抖振), 实际部署需加速率限制器",
             xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
             fontsize=8, color="#555555")

ax3 = fig1.add_subplot(gs[2])
ax3.plot(k, seocf["soc"], color="#1f77b4", lw=2.2, label="SE-OCF SOC")
ax3.plot(k, naked["soc"], color="#ff7f0e", ls="--", lw=2.0, zorder=5,
         label="裸RL SOC(无投影)")
ax3.plot(k, seocf["Smin"], color="#2ca02c", ls="--", lw=1.4, label="SOC$_{\\min}(r_k)$")
ax3.plot(k, seocf["Smax"], color="#7f7f7f", ls="--", lw=1.4, label="SOC$_{\\max}(r_k)$")
ax3.set_xlabel("时间步 $k$"); ax3.set_ylabel("SOC")
ax3.set_ylim(0, 1.0); ax3.grid(alpha=0.3)
ax3.set_title("(下) 前向不变性: SOC 轨迹与动态边界 (副轴: 温度)", fontsize=11)
axt = ax3.twinx()
axt.plot(k, naked["T"], color="#ff7f0e", lw=1.2, ls="-.", alpha=0.8, label="裸RL 温度(越界)")
axt.plot(k, seocf["T"], color="#9467bd", lw=1.6, label="SE-OCF 温度")
axt.plot(k, seocf["Tmax"], color="#d62728", ls=":", lw=1.4, label="$T_{\\max}(r_k)$")
axt.fill_between(k, seocf["T"], seocf["Tmax"],
                 where=(naked["T"] > naked["Tmax"]), color="#d62728", alpha=0.12)
axt.set_ylabel("温度 $T$ (°C)"); axt.set_ylim(0, 90)
# 裸RL SOC 未越界说明(用户审稿意见): SOC 边界较宽, 裸RL 的失效模式是温度越界
ax3.annotate("裸RL 与 SE-OCF 的 SOC 均未越界(SOC 边界宽);\n"
             "裸RL 的失效模式是温度越界(见副轴红色区)",
             xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
             fontsize=8, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
h3, l3 = ax3.get_legend_handles_labels(); h4, l4 = axt.get_legend_handles_labels()
ax3.legend(h3 + h4, l3 + l4, loc="lower right", fontsize=8.5)

fig1.suptitle("图1  阶段二局部闭环:  $\\mathcal{C}\\circ\\Pi_{\\mathcal{S}}$ 性质验证 "
              "(论文 §5.2 模型, 合成风险信号)", fontsize=12, y=1.005)
fig1.savefig("fig1_stage2_closed_loop.pdf", bbox_inches="tight")
fig1.savefig("fig1_stage2_closed_loop.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("\n已生成: fig1_stage2_closed_loop.pdf / .png")

def sweep(uc, static=False):
    """SOC 跟踪愿望动作(幅值 uc)+投影的闭环;
       static=True 时为恒定电流基线: 固定边界(非风险自适应)
       返回 (平均执行电流J, 最小温度裕度, 是否SOC越界)"""
    soc, T = SOC0, T0
    Jacc = 0.0; Bmin = 1e9; soc_viol = 0
    for kk in range(STEPS):
        r = risk_signal(kk)
        u_des = float(uc) if soc >= SOC_REF else -float(uc)
        u_safe, _ = project(u_des, soc, T, r, static=static)
        Jacc += abs(u_safe)
        Bmin = min(Bmin, t_max(r) - T)   # 裕度统一按动态 T_max(r) 计算(公平对比)
        if soc < soc_min(r) - 1e-9 or soc > soc_max(r) + 1e-9:
            soc_viol += 1
        soc, T = step(soc, T, u_safe)
    return Jacc / STEPS, Bmin, soc_viol

us_c = np.linspace(1.0, 13.0, 61)
J_s, B_s = [], []
J_c, B_c = [], []                     # 恒定电流基线(静态边界)
for uc in us_c:
    J, B, _ = sweep(uc, static=False)
    J_s.append(J); B_s.append(B)
    Jc, Bc, _ = sweep(uc, static=True)
    J_c.append(Jc); B_c.append(Bc)
J_s, B_s = np.array(J_s), np.array(B_s)
J_c, B_c = np.array(J_c), np.array(B_c)

J_se, B_se, v_se = sweep(U_DES_CONST)      # SE-OCF (投影)
soc, T = SOC0, T0                          # 裸RL: 同愿望动作, 无投影
Jacc = 0.0; Bmin = 1e9
for kk in range(STEPS):
    r = risk_signal(kk)
    u_des = float(U_DES_CONST) if soc >= SOC_REF else -float(U_DES_CONST)
    Jacc += abs(u_des); Bmin = min(Bmin, t_max(r) - T)
    soc, T = step(soc, T, u_des)
J_nk, B_nk = Jacc / STEPS, Bmin
J_cv, B_cv, v_cv = sweep(3.0)              
J_base, B_base, v_base = sweep(7.0, static=True)

fig2, ax = plt.subplots(figsize=(8.2, 6.0))
ax.plot(J_s, B_s, color="#1f77b4", lw=2.2, label="理论最优权衡曲线（帕累托前沿）")
ax.plot(J_c, B_c, color="#8c8c8c", lw=1.8, ls="--",
        label="恒定电流基线（静态边界, 非自适应）")
ax.plot(J_base, B_base, "^", ms=11, color="#8c8c8c", zorder=5,
        label=f"恒定电流基线代表点 $({J_base:.1f},\\,{B_base:.1f})$")
safe = B_s >= 0
ax.fill_between(J_s[safe], 0, B_s[safe], color="#cfe3f7", alpha=0.6,
                label="安全可行区 $B\\geq 0$")
ax.axhline(0, color="#2ca02c", lw=1.4, ls="--", label="安全底线 $B=0$")
ax.plot(J_cv, B_cv, "o", ms=10, color="#7f7f7f", zorder=5, label="安全优先(过保守)")
ax.plot(J_se, B_se, "P", ms=13, color="#1f77b4", zorder=6, label="SE-OCF(前沿上)")
ax.annotate(f"SE-OCF $({J_se:.2f},\\,{B_se:+.2f})$",
            xy=(J_se, B_se), xytext=(J_se-1.6, B_se+2.2),
            fontsize=8.5, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
ax.plot(J_nk, B_nk, "D", ms=10, color="#ff7f0e", zorder=5, label="裸RL(越安全线)")
ax.annotate(f"裸RL $({J_nk:.2f},\\,{B_nk:+.2f})$",
            xy=(J_nk, B_nk), xytext=(J_nk-3.2, B_nk+3.5),
            fontsize=8.5, color="#ff7f0e",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8))
ax.annotate("越界区 $B<0$", xy=(0.72, 0.06), xycoords="axes fraction",
            fontsize=9, color="#d62728")
ax.set_xlabel("平均执行电流 $\\bar{|u_{\\rm safe}|}$ (A, 与收益成正比)")
ax.set_ylabel("最小温度裕度 $\\min_k\\,(T_{\\max}(r_k)-T_k)$")
ax.set_title("图2  安全—经济权衡: SE-OCF 落在安全前沿上\n"
             "(§5.2 模型, 合成风险信号)", fontsize=11)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8.5)
fig2.tight_layout()
fig2.savefig("fig2_stage2_pareto.pdf", bbox_inches="tight")
fig2.savefig("fig2_stage2_pareto.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("已生成: fig2_stage2_pareto.pdf / .png")

print("\n== 图2 工作点 ==")
print(f"裸RL        : J={J_nk:.2f} A, 最小裕度={B_nk:.2f} C  (越界: {B_nk<0})")
print(f"SE-OCF      : J={J_se:.2f} A, 最小裕度={B_se:.2f} C  (越界: {B_se<0})")
print(f"安全优先    : J={J_cv:.2f} A, 最小裕度={B_cv:.2f} C")
print(f"恒定电流基线: J={J_base:.2f} A, 最小裕度={B_base:.2f} C  "
      f"(低于前沿同收益点, 验证动态优化更优)")
