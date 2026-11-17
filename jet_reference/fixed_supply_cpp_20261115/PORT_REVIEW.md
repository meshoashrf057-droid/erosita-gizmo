# Isolated fixed-external-supply CPP port review01

This directory contains a source preparation, not a compiled or runtime-validated jet. No original CPP/C source, Config, GIZMO binary, orbit file, or campaign state was edited. No jet IC, compilation, smoke, or production was executed here. The ordered user workflow is orbit acceptance → all orbit snapshot figures/video → owned endpoint angular-momentum adjustment → third-stage eROSITA.

## Sources and scope

The immutable base is /home/xingdi/gizmo-public-gizmo-cpp. The comparison reference is /home/xingdi/gizmo-public, especially galaxy_sf/blackholes/blackhole.c, blackhole_environment.c, blackhole_swallow_and_kick.c, init.c, timestep.c, begrun.c and allvars.h. The historical physical controls come from /home/xingdi/gizmo-runs/erosita_xray/erosita.param; the current gas/cooling/star-formation/gravity/numerical controls must come from the accepted n3000 orbit production parameters, not from that older param wholesale.

source_input_manifest.json records 339 copied ordinary files and excluded .git, generated/configuration products, compiled products, logs, and Zone.Identifier streams. source_port_manifest.json records every prepared source hash and the exact nine modified files. fixed_supply_cpp.patch is the complete original-to-copy diff. The copy script's strict unique-pattern guard initially rejected two similar environment loops; it was narrowed to the main loop and only the remaining edits were applied. Both manifests retain this history. Original copied-source hashes and protected original Config/binaries were read back unchanged.

## Behavioral mapping

| Legacy C | Isolated CPP | Preserved meaning |
| --- | --- | --- |
| BLACK_HOLES | SINK_PARTICLES | Explicit sink module, without FIRE_BHS or swallowing/accretion presets |
| BH_REPOSITION_ON_POTMIN=2 | SINK_REPOSITION_ON_POTMIN | Existing CPP smooth potential-minimum repositioning path |
| BH_DEBUG_FIX_MDOT_MBH=(0.00124) | SINK_DEBUG_FIX_MDOT_MBH=(0.00124) | Override local supplied dm, in code mass/code time; not Mdot/Mbh and not multiplied by BH mass |
| BH_fb_duty_cycle/BH_fb_period | Sink_fb_duty_cycle/Sink_fb_period | Registered real controls; legacy aliases accepted only for these two |
| BH_WIND_SPAWN=2 | SINK_WIND_SPAWN=2 | Native paired mass-reservoir wind spawning |
| BH_FOLLOW_ACCRETED_ANGMOM=1 | SINK_FOLLOW_ACCRETED_ANGMOM=1 | Native evolving sink angular-momentum field |
| BH_FB_COLLIMATED | SINK_FB_COLLIMATED | Native collimation along sink angular momentum |
| BH_FOLLOW_ACCRETED_COM/MOMENTUM | SINK_FOLLOW_ACCRETED_COM/MOMENTUM | Explicit flags retained; native source suppresses their position/momentum-following updates when repositioning is enabled |
| BH_INTERACT_ON_GAS_TIMESTEP | SINK_INTERACT_ON_GAS_TIMESTEP | Existing accumulated gas-search dt, kept unchanged |
| BH_OUTPUT_MOREINFO | SINK_OUTPUT_MOREINFO | Native diagnostics |
| BH_WAKEUP_GAS/LowestBHTimeBin | SINK_WAKEUP_GAS/LowestSinkTimeBin | Per-gas minimum interacting sink bin, initialization, imported/local search transfer, cap and reset |
| Existing legacy reservoir | OUTPUT_UNSPAWNED_SINKMASS | Required to round-trip unspawned wind mass via native CPP HDF5 input/output |

For continuous duty >= 1, supplied dm = 0.00124 × dt. For 0 < duty < 1, period_code = period_Gyr / UNIT_TIME_IN_GYR, on_time = duty × period_code; when fmod(All.Time, period_code) < on_time, dm = 2 × (0.00124/duty) × sin²(pi × All.Time/on_time) × dt, otherwise zero. This preserves the legacy absolute All.Time phase, including the sine's legacy behavior across periods for nonreciprocal duties. It does not reset phase at the new jet start or at each period. Nonfinite/nonpositive duty or period now fails closed before integration. The authorized stage uses duty=period=1, so the continuous branch is selected.

Exactly three active mass-debit sites are exempted under the fixed-supply macro: radiative subtraction from dynamical/subgrid BH mass; spawned-wind subtraction from subgrid BH mass; spawned-cell subtraction from dynamical BH mass. The unspawned reservoir is still decreased for every spawned cell. The native wind ratio (1-f)/f and existing per-step mass caps are unchanged. Sink_Mdot is not overwritten and may correctly remain zero while externally supplied jets spawn. No naturally accreted gas or swallowed particles fund this supply.

## Gas timestep boundary audit

Old C allvars.h:939–940 and CPP declarations/macros.h:13–14 use the identical GET_INTEGERTIME_FROM_TIMEBIN and GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN formula. Bin zero explicitly maps to zero, not one. Old C chooses 60 bins under LONG_INTEGER_TIME, otherwise 29; native CPP declarations/typedefs.h:3–6 always uses long long and 60 bins. The port uses TIMEBINS symbolically, never a literal 29/60.

The unset sentinel is TIMEBINS; the port skips bins outside [0,TIMEBINS). For a valid remembered bin, it applies the legacy dt > 2×sink_dt condition, dt = .99×(2×sink_dt), then resets the remembered bin to TIMEBINS. If the cap is not applied, the recorded bin remains. Bin zero retains the legacy zero-cap behavior and must be monitored in runtime testing; it was not silently redefined.

The main environment neighbor loop sends the sink's TimeBin and takes the minimum for gas only. A named OpenMP critical region protects both comparison and assignment, replacing the legacy comparison plus separate atomic write that did not guarantee a race-safe minimum. OMP1 gas behavior is preserved. Non-gas particles never consume this field. Gas initialization sets the sentinel on normal/flag2 initialization. The field is transient, not a new HDF5 input block. A new binary restart includes the new struct, so old orbit/C restarts must never be fed to this new executable.

## Stage inputs and controls

stage3_contract.json records the exact 30-physical-Myr timeline, all thirteen legacy-to-CPP parameter names, supply budget, and preserved n3000 controls. sink_input_schema.json records the six required new PartType5 fields and dtype evidence.

The accepted endpoint already has one BH ID 90000000, dynamical Masses=.0004, but none of the native CPP Sink_* datasets. The complete copied endpoint is therefore required before a flag2 restart with this build; azaxis alone is insufficient. The azaxis correction only sets Sink_Specific_AngMom to [0,0,1] in code units. It does not rotate all particles or realign the galaxy, and does not impose a permanent fixed axis. Initializing a previously absent state is distinct from reinitializing a future jet restart; subsequent actual reservoir and sink fields must be preserved.

The new exact TimeMax is 1.6670206029274603 for TimeBegin 1.6363392421373844 and 30 physical Myr. The older POST_T13_PIPELINE .03-code target is superseded by root AGENTS' later explicit physical-30-Myr clarification. Current CritPhysDensity=3000, MaxKernelRadius=1e56, MinSizeTimestep=1e-12 and all nonjet physics are retained.

No source-derived input schema is a substitute for inspecting the actual generated GIZMO_config.h and native blockpresent() under the final compiler configuration. At that gate, verify all enabled sink blocks, cooling tables and parameter names against the full flag2 input. Canonical Sink* parameters must replace the old BlackHole*/BAL_* lines; original CPP does not register those old names, except the two new duty/period aliases.

## Suggested later build command (NOT executed)

From the existing attached, visible Ubuntu terminal, after the independent source review:

    cd /home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01/source
    set -o pipefail
    nice -n 19 make -j2 CONFIG=Config.sh EXEC=GIZMO_fixed_supply_cpp 2>&1 | tee ../build_terminal.log

The stock Makefile.systype selects github-ubuntu. mpicc, mpicxx, make, the historical HDF5 header/archive under /home/xingdi/gizmo-public/hdf5-1.8.21/hdf5, GSL headers/library, and FFTW headers/library were observed present. No PATH/account/proxy change or package install is suggested. Make's config-makefile generates GIZMO_config.h/compile_time_info.cc inside this owned copy. Record complete build output, return code, compiler versions, generated configuration and binary SHA after successful compilation. Never substitute the old orbit binary.

## Outstanding runtime gates

1. Compile in the visible terminal and independently inspect enabled/disabled preprocessor flags, symbol/parameter support, complete HDF5 block requirements, binary/Config/source hashes and unchanged n3000 physics.
2. Produce an owned full endpoint copy, preserve every old dataset/Header and identity, then add the six sink fields. Validate the full static inventory, finite values, metal/internal-energy/stellar-age/identity invariants, units and exact times before flag2.
3. Use a separate visible flag2 smoke with 6 MPI, OMP1, nice19 and full logs. A tiny orbit-style smoke cannot establish spawning: at the requested rate a pair requires about 0.157709 Myr of supply. A 0.2–0.3 physical-Myr smoke is a suggested practical window, not an already-approved runtime result. Check actual paired spawn events, evolving residual reservoir, finite fields, unchanged BH dynamical/subgrid masses, bipolar velocity geometry and identity/generation handling. Watch bin-zero or minimum-step failures. Sink_Mdot=0 by itself is expected, not a smoke failure.
4. Check external mass conservation as cumulative spawned gas plus remaining reservoir against supplied mass, including actual start/end dt and any legacy caps. Do not demand a compensating BH mass decrease. Preserve mass/metal/star-age checks for pre-existing particles separately.
5. Require independent smoke review and production clearance before the single 30-Myr launch from the reviewed full jet IC. Never use the smoke endpoint as an unreviewed production replacement or reuse the old orbit binary restart.

static_checks.json/log cover source/text integrity and synthetic Python boundary examples only. They do not establish compile success, numerical stability, jet morphology, or runtime scientific acceptance.
