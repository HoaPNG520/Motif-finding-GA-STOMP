"""Writes fake ga_stomp output dirs (same CSV formats as the binary) so CI can
smoke-test plot_results.py without a GPU.  Usage: python tests/make_mock_results.py OUT_DIR
"""
import numpy as np, os, sys
rng=np.random.default_rng(0)
def make(d, period, n=800, synth=False):
    os.makedirs(d, exist_ok=True)
    t=rng.normal(0,0.3,n)
    for s in range(50, n-40, period): t[s:s+30]+=np.sin(np.linspace(0,6*np.pi,30))*np.exp(-np.linspace(0,3,30))*3
    m=period; L=n-m+1
    S=np.lib.stride_tricks.sliding_window_view(t,m); Z=(S-S.mean(1,keepdims=True))/S.std(1,keepdims=True)
    D=np.sqrt(np.maximum(2*m*(1-(Z@Z.T)/m),0)); ez=m//4
    for i in range(L): D[i,max(0,i-ez):i+ez+1]=np.inf
    mp=D.min(1); mpi=D.argmin(1)
    open(f"{d}/series.csv","w").write("index,value\n"+"".join(f"{i},{v:.6f}\n" for i,v in enumerate(t)))
    open(f"{d}/matrix_profile.csv","w").write(f"# Matrix Profile output\n# window_size = {m}\n# profile_len = {L}\nsubseq_index,mp_distance,mp_index\n"+"".join(f"{i},{a:.6f},{b}\n" for i,(a,b) in enumerate(zip(mp,mpi))))
    g=np.arange(20); best=0.9-0.4*np.exp(-g/4); mean=best-0.15-0.1*np.exp(-g/5)
    open(f"{d}/convergence.csv","w").write("generation,best_fitness,mean_fitness\n"+"".join(f"{a},{b:.6f},{c:.6f}\n" for a,b,c in zip(g,best,mean)))
    ms=period+rng.integers(-4,5,3)
    open(f"{d}/multiseed.csv","w").write("seed,best_m,best_ez,best_k,best_fitness,wall_ms\n"+"".join(f"{42+1000*i},{v},0.3,5,0.85,12000\n" for i,v in enumerate(ms))+f"MEAN,{ms.mean():.2f},0.3,,0.85,12000\nSTD,{ms.std():.2f},0,,0,0\n")
    open(f"{d}/timing.csv","w").write("metric,value_ms\ntotal_wall,15000\nstomp_total,11000\nprecompute_total,900\nfitness_total,2600\nga_overhead,40\nn_stomp_calls,400\navg_per_stomp,27.5\n")
    open(f"{d}/summary.csv","w").write("input,n,best_m,best_ez,best_k,fitness,contrast,regularity,period_reward,detected_period,band_min,band_max,sampling_rate,top_motif_idx,true_m,true_pos1,wall_ms\n"+f"{d},{n},{period},0.3,5,0.87,0.4,0.8,0.98,{period+2},40,240,12000,{mp.argmin()},{80 if synth else 0},{200 if synth else -1},15000\n")
    mm=np.arange(20,201,5)
    reg=np.clip(1-np.abs(mm-30)/60,0,1); con=0.2+0.1*np.sin(mm/20); per=np.exp(-0.5*((mm-period)/(0.2*period))**2)
    comp=0.2*con+0.3*reg+0.5*per
    nul=0.15+0.0015*mm
    open(f"{d}/landscape.csv","w").write(f"# detected_period = {period+2}\n# null = phase\nm,composite,contrast,regularity,period_reward,contrast_raw,contrast_null,regularity_null\n"+"".join(f"{a},{b:.4f},{c-n:.4f},{e:.4f},{g:.4f},{c:.4f},{n:.4f},nan\n" for a,b,c,e,g,n in zip(mm,comp,con,reg,per,nul)))
    if synth:
        mm=np.arange(30,200,10); fit=np.exp(-((mm-80)/30)**2); rec=(abs(mm-80)<=20).astype(float)
        open(f"{d}/sweep.csv","w").write("# Spearman rho = 0.8123\nm,fitness,recall,top_dist,regularity,consistency,contrast\n"+"".join(f"{a},{b:.4f},{c:.1f},1,1,1,1\n" for a,b,c in zip(mm,fit,rec)))
make(sys.argv[1]+"/IR007",74); make(sys.argv[1]+"/OR007",112); make(sys.argv[1]+"/synthetic",80,synth=True)
for v, shift in [("full",0),("no_period",-40),("no_regularity",3),("no_contrast",1)]:
    make(f"{sys.argv[1]}/ablation/{v}/IR007",max(20,74+shift)); make(f"{sys.argv[1]}/ablation/{v}/OR007",max(20,112+shift))

# detection.csv in the fault_detection layout (one row per segment)
rows=["name,cls,size,load,segment,expected_m,best_m,kurtosis,rms,contrast_raw,contrast_corr,land_raw_mean,land_corr_mean"]
for cls in ("Normal","IR","OR","B"):
    for load in range(4):
        for seg in range(3):
            e={"IR":75,"OR":113}.get(cls,"")
            b=(e+int(rng.integers(-3,4))) if e else 150
            f=0 if cls=="Normal" else 1
            rows.append(f"{cls}_{load}hp_s{seg},{cls},{7*f},{load},{seg},{e},{b},{rng.normal(3+10*f,1):.3f},{rng.normal(0.1+0.2*f,0.02):.3f},{rng.uniform(0.1,0.4):.3f},{rng.normal(0.15*f,0.03):.3f},{rng.uniform(0.1,0.4):.3f},{rng.normal(0.15*f,0.03):.3f}")
os.makedirs(sys.argv[1]+"/detection",exist_ok=True)
open(sys.argv[1]+"/detection/detection.csv","w").write("\n".join(rows)+"\n")

# band_compare layout: IR/OR rows for two variants
rows=["name,variant,cls,size,load,rpm,expected_m,best_m,detected_period,band_min,band_max"]
for v,(lo,hi) in (("band_50_300",(40,240)),("band_geometry",(67,130))):
    for cls,order in (("IR",5.4152),("OR",3.5848)):
        for size in (7,14,21):
            for load,rpm in enumerate((1797,1772,1750,1730)):
                e=round(12000/(order*rpm/60)); b=e+int(rng.integers(-3,4))
                rows.append(f"{cls}{size:03d}_{load}hp,{v},{cls},{size},{load},{rpm},{e},{b},{b},{lo},{hi}")
open(sys.argv[1]+"/detection/band.csv","w").write("\n".join(rows)+"\n")
