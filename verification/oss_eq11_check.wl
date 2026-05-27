(* ::Package:: *)

(* Independent Wolfram Language cross-verification of the OSS process
   (Kato & Shigeta 2020, eq. (10)-(11)) implemented in src/oss.rs.

   It reimplements the OSS index structure and the three OssWeight variants
   from scratch and reproduces:

     (A) the deterministic 8-channel factors printed by examples/xcheck_oss and
         pinned in tests/processes.rs (oss_sample_std_matches_paper_eq11_reference);
     (B) the 200-trial Poisson Monte Carlo behind the F2 decision (which weight
         best recovers the gains under realistic noise).

   The Rust `OssWeight::SampleStd` (paper eq. 11: sigma_k = sample std of the
   local factors c_OSS_k over the channels sharing overlap level k) agrees with
   the "SampleStd" branch below to ~1e-15 on case (A); `PoissonPropagated` and
   `OverlapCount` likewise match their Rust counterparts.

   Run:  wolframscript -f verification/oss_eq11_check.wl
   (Originally executed via the Wolfram MCP evaluator.)
*)

(* ---- shared OSS index structure (0-based q/measurement, matching src/oss.rs) ----
   At overlap level s (1..2*nb-3) the contributing blocks are q in
   [max(0, s-(nb-1)), min(s, nb-1)]; channel pos+q*block reads at step s-q.
   Here block = 1 (so channel == q) and a single within-block position. *)
qrange[s_, nb_] := Range[Max[0, s - (nb - 1)], Min[s, nb - 1]];

(* Global OSS factors from a measurement matrix Y[[m+1, c+1]] (0-based m, c)
   for one weight mode. Mirrors optimized_single_step in src/oss.rs. *)
ClearAll[ossFactors];
ossFactors[Y_, nb_, mode_] := Module[
  {last = 2 nb - 3, wsum = ConstantArray[0., nb], wf = ConstantArray[0., nb],
   qs, vals, cnt, ref, locs, sd, w},
  Do[
    qs = qrange[s, nb];
    vals = N@Table[Y[[s - q + 1, q + 1]], {q, qs}];
    cnt = Length[qs]; ref = Mean[vals]; locs = ref/vals;
    (* eq. (11): sigma_k is the sample std (divisor n-1) of the local factors
       over this level -> one weight 1/sigma_k^2 shared by every channel. *)
    sd = If[cnt >= 2, StandardDeviation[locs], 0.];
    Do[
      With[{v = vals[[k]], c = locs[[k]], q = qs[[k]]},
        w = Switch[mode,
          "SampleStd",         If[sd > 0, 1./sd^2, 0.],
          "PoissonPropagated", 1./(ref/(cnt v^2) + ref^2/v^3),
          "OverlapCount",      N[cnt]];
        If[w > 0, wsum[[q + 1]] += w; wf[[q + 1]] += w c]],
      {k, cnt}],
    {s, 1, last}];
  Table[If[wsum[[i]] > 0, wf[[i]]/wsum[[i]], 1.], {i, nb}]];

(* SS reference: the single full-overlap level s = nb-1 (count nb). *)
ssFactors[Y_, nb_] := Module[{s = nb - 1, qs = qrange[nb - 1, nb], vals, ref},
  vals = N@Table[Y[[s - q + 1, q + 1]], {q, qs}]; ref = Mean[vals];
  Table[ref/N[Y[[s - (i - 1) + 1, i]]], {i, nb}]];

(* ---- (A) deterministic 8-channel noiseless case -------------------------- *)
(* Expected (matches Rust to ~1e-15):
     SampleStd:        {1.019590268391508, 0.925985222505392, 1.118115807949738,
                        0.959162871302379, 1.057813959519594, 0.839098869525341,
                        1.238079866158647, 0.969368643187109}
     PoissonPropagated:{1.012548018431703, 0.919252376737347, 1.116039772698576,
                        0.957022611177646, 1.055277755683186, 0.836082427923770,
                        1.235463924831642, 0.967737952670133}
     OverlapCount:     {1.010571428571428, 0.917532467532468, 1.116183574879227,
                        0.956944444444444, 1.055921052631579, 0.836413043478261,
                        1.243750000000000, 0.974509803921568}                   *)
gDet = {1.00, 1.10, 0.90, 1.05, 0.95, 1.20, 0.80, 1.02}; IDet = 1000.; nbDet = 8;
(* noiseless flat scatterer: every (m, c) entry is gain[c]*I *)
YDet = Table[gDet[[c]] IDet, {m, nbDet}, {c, nbDet}];
Print["=== (A) deterministic 8-channel factors (cf. examples/xcheck_oss) ==="];
Do[Print[mode, ":\n  ", NumberForm[ossFactors[YDet, nbDet, mode], 16]],
  {mode, {"SampleStd", "PoissonPropagated", "OverlapCount"}}];

(* ---- (B) 200-trial Poisson Monte Carlo (nb=32, I=1e6, 1% gain spread) ----- *)
(* Expected mean corrected interior TFU %  {SS, SampleStd, PoissonProp, OverlapCount}
     ~ {0.1391, 0.1332, 0.1334, 0.1158};  Poisson floor 0.10%.
   fraction OSS<=SS {SampleStd, PoissonProp, OverlapCount} ~ {0.585, 0.58, 0.91}.
   Reading: under noise SampleStd (paper) ~= PoissonPropagated, both below SS;
   OverlapCount is lowest on a flat field because it favours the full-overlap
   level, minimising the subset-normalisation bias of the partial levels.       *)
SeedRandom[20260527];
nb = 32; Ival = 1.*^6; spread = 0.01; trials = 200; edge = 2;
gtrue = Table[Max[1 - 3 spread, 1 + spread RandomVariate[NormalDistribution[]]], {nb}];
noisyY[] := N@Table[RandomVariate[PoissonDistribution[gtrue[[c]] Ival]], {m, nb}, {c, nb}];
tfuI[v_] := Module[{x = v[[edge + 1 ;; nb - edge]]}, N[100 StandardDeviation[x]/Mean[x]]];
res = Table[
  Module[{Y = noisyY[], samp},
    samp = N@Table[RandomVariate[PoissonDistribution[gtrue[[c]] Ival]], {c, nb}];
    {tfuI[samp ssFactors[Y, nb]],
     tfuI[samp ossFactors[Y, nb, "SampleStd"]],
     tfuI[samp ossFactors[Y, nb, "PoissonPropagated"]],
     tfuI[samp ossFactors[Y, nb, "OverlapCount"]]}],
  {trials}];
Print["=== (B) 200-trial Poisson MC (nb=32, I=1e6, 1% gain spread) ==="];
Print["mean corrected interior TFU %  {SS, SampleStd, PoissonProp, OverlapCount}: ", Mean[res]];
Print["fraction OSS<=SS {SampleStd, PoissonProp, OverlapCount}: ",
  Table[N@Mean[Boole[#[[j]] <= #[[1]]] & /@ res], {j, 2, 4}]];
Print["Poisson floor %: ", N[100/Sqrt[Ival]]];
