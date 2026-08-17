# Luma Swarm Studio



https://github.com/user-attachments/assets/c04d0072-edc0-47b7-afd2-4c4d368185a1



**Version 1.0**

Luma Swarm Studio is a native, cinematic 3D drone-show simulator and choreography editor built in Rust with `wgpu`, WGSL compute/render shaders, and `egui`. It targets macOS and Apple Silicon first while using portable APIs that also work on Windows and Linux.

The project is a complete vertical slice: run a continuously looping default choreography, scrub or edit its timeline, preview twelve procedural formations, turn PNG/JPEG artwork into a launch/hold/land show, play an animated GIF between launch and landing, inspect replayable collision records, tune the independent flight model and weather, switch between flight-validated and GPU-resident execution, navigate with orbit or first-person free-fly cameras, and enter full-screen presentation mode.

## Run

Install a current stable Rust toolchain, then from this directory:

```bash
cargo run --release
```

The first build downloads the Rust dependencies. Development builds work too (`cargo run`) but release mode is recommended for fleets above roughly 1,000 drones.

## Controls

- `Space`: play or pause.
- `F11`: enter or leave full-screen presentation mode.
- `Escape`: release free-fly mouse capture, then leave presentation mode.
- Drag in the viewport: switch to orbit camera and rotate.
- Scroll in the viewport: zoom.
- Free-fly: select `Free-fly`, click the viewport to capture the mouse, use `WASD` to move (`A` strafes left and `D` strafes right), `Q/E` vertically, hold `Shift` to sprint, and use the wheel to adjust movement speed. Horizontal mouse look uses an inverted X axis.
- Click a formation card: jump to its first cue and pause for inspection.
- Timeline: scrub, select, duplicate, or reorder cues.
- Show inspector: choose the camera, frame the entire fleet, drop a PNG/JPEG/GIF (or paste its path), and set the still hold or GIF playback duration.
- Fleet inspector: switch execution mode; tune flight limits, stabilisation, separation, wind, and fleet size (100–5,000 validated or up to 1,000,000 GPU-resident); enable/disable GPU collision correction; enable or disable the independent full-fleet GPU safety audit; toggle drone-separation monitoring; independently show or hide amber/red visual alerts; and inspect the cumulative run collision log. Ground contact remains monitored whenever telemetry is active; altitude/height envelopes are deliberately not monitored.
- Look inspector: choose Low, Medium, High, or Cinematic and tune HDR exposure, bloom, saturation, haze, light size, reflections, and render scale.

GPU-resident execution is selected automatically when the device exposes the required compute and storage-buffer limits, with a default fleet of 20,000. Unsupported devices fall back to flight-validated execution with 5,000 drones. Switching modes restores those respective mode defaults. Wind defaults to zero.

## What is simulated

Every drone carries independent position, previous position, velocity, acceleration, orientation, formation slot, RGBW output, brightness, animation phase, rotor angle, and battery state. The simulation runs at a fixed 60 Hz and rendering interpolates between fixed steps.

Shows remain deterministic because formations and transition paths are generated from timeline time rather than emergent boid behavior. Smaller fleets use exact Hungarian assignment; larger fleets use Morton spatial ordering with a local improvement pass. Transitions use zero-endpoint-velocity quintic curves, vertical clearance arcs, deterministic lateral lanes, staggered take-off/landing waves, and a spatial hash for local separation correction.

Flight-validated mode audits every fixed simulation step. Its drone-separation monitor exhaustively checks candidate pairs within a 0.38 m warning volume against a 0.24 m physical collision envelope, while ground contact is always checked. It does not apply an altitude/height envelope. The audit records incident episodes and exposes current drone/ground clearance. A run-level log separately accumulates newly occurring collision pairs and ground contacts without counting one persistent overlap on every frame. Each entry retains run time, show time, representative drone IDs, world-space coordinates and measured clearance; `GO TO` seeks to the exact audited timeline position and `REPLAY` starts one second earlier. Visual alert colouring is independent: it can be hidden without disabling correction or telemetry. A position-based constraint stage projects neighbours beyond the warning volume after ordinary steering and target tracking. Its spatial buckets, candidate pairs, flags and correction offsets are retained between steps, and live correction is deliberately bounded so a slow frame cannot trigger a recursive catch-up spiral. Editor seeks are explicitly treated as teleports and do not fabricate incident history.

The built-in formations are:

- Stellar chrysalis with twelve animated light ribbons, a pulsing golden core, and tilted orbit halos
- Layered neon heart
- Spiral galaxy
- Prism cathedral with radial gothic vaults, a rotating rose window, and rising spectral spires
- Chromatic DNA helix
- Ringed planet with animated atmosphere, storms, layered rings, and orbiting moons
- Braided infinity portal
- Opening prismatic lotus
- Rotating celestial crown
- Bonus Event horizon with a differential accretion disc, photon sphere, and braided polar jets
- Bonus Spectral mandala with eighteen opening light petals, offset halo rings, and a radiant core
- Bonus Chrono gyroscope with eight independently tilted toroidal gates and a pulsing temporal core

`image_formation::sample` treats every uploaded PNG/JPEG as a complete rectangular raster: there is no segmentation, background removal, alpha masking, or subject crop. It Lanczos-downsamples large sources to a sharp aspect-correct grid that never contains more raster cells than available drones, so every downsampled pixel receives a slot and random-looking holes cannot appear. The small remainder is distributed behind existing pixels at a safe depth. Source sRGB is converted to linear emission without artificial saturation. Upload rendering gives every drone four independently coloured R, G, B, and W emitters; their additive output reconstructs the sampled source colour while supplying four visible light elements per aircraft.

Animated GIFs use the same fixed, safety-spaced lattice. Only the RGBW values change from frame to frame, so playback creates no assignments, movement, or new collision opportunity. The GIF plays once during an adjustable timeline cue, with its first frame used for launch and its last frame retained for landing. Very long animations are temporally compacted to a bounded set of frames while preserving total playback duration. The same media path works in both execution modes. `formation_import::sample_obj` also samples OBJ polygon surfaces (or vertex-only point clouds) into the normalized RGBW format.

## Rendering pipeline

The viewport is a custom `egui_wgpu` callback. Each frame:

1. Flight-validated CPU state is packed into a data-oriented instance buffer, or GPU mode generates formation/assignment/correction state directly in the compute pass.
2. A WGSL compute pass prepares pulse-corrected render instances and deterministic high-count morphs.
3. A single instanced draw renders detailed quadcopter bodies and rotors into an `RGBA16Float` HDR target.
4. A second instanced draw renders additive distance-scaled RGBW light billboards.
5. The field, launch grid, stadium ring, depth, stars, haze, ground response, and cloud layer establish the environment.
6. A final fullscreen pass adds bloom, exposure, hue-preserving highlight compression, adjustable saturation, and vignette before compositing into the editor.

This keeps draw-call count nearly constant as the fleet grows. Light/body detail naturally reduces with distance because geometry becomes subpixel while light billboards remain readable. Render scale is independent from simulation quality.

### Synchronization and scale

Formation transitions are timeline-locked. Animated geometry is evaluated from one continuous show clock across morph/animate cue boundaries, so changing cue type cannot reset animation phase. Each aircraft receives the planned trajectory's velocity and acceleration as feed-forward inputs, then applies its own stabilisation, wind rejection and separation correction around that plan. A progressive endpoint lock absorbs residual disturbance before the cue boundary, so increasing fleet size does not create an ever-later tail of aircraft. Large-fleet assignments normalize both point clouds before Morton ordering and refine routes at multiple spatial scales; adjacent timeline cues that keep the same formation also preserve their existing slot assignment.

The editor exposes up to 5,000 fully modelled, individually audited aircraft in Flight validated mode. GPU resident mode is a separate choreography and visualisation tier: built-in formations, deterministic index assignment, morph paths, and phase correction are produced in WGSL without allocating one CPU vehicle per virtual aircraft. It accepts up to 1,000,000 virtual drones, drops detailed body meshes above 50,000, and keeps light impostors GPU-resident.

GPU choreography is safety-spaced by default. Surface area grows in proportion to fleet size with an additional operational margin, high-count procedural formations use regular rather than randomly clustered parameterizations, and transitions climb vertically into separated flight levels before crossing and descending. A separately toggleable GPU correction stage performs eight to eighteen density-aware ping-pong spatial-hash passes before rendering; it reads one instance buffer and writes another, avoiding cross-thread position races.

GPU safety certification is enabled by default and remains an independent performance toggle. When enabled, the corrected rendered positions are inserted into a GPU-resident spatial hash. Ground contact is always checked; the drone-separation monitor adds the exhaustive 27-cell-neighbour audit for the 0.38 m warning and 0.24 m collision envelopes. Height limits are not audited. Counters, audited show time, representative drone IDs and incident coordinates are read back through a staged asynchronous ring and feed the same cumulative run log as CPU mode. Amber/red recolouring is separately toggleable and never changes the audit result or correction path.

## macOS application

Build or refresh the self-contained, ad-hoc signed Mac application with:

```bash
./scripts/package_macos.sh
```

This always compiles the current release source, embeds the complete multi-resolution `assets/AppIcon.icns` generated from `assets/app-icon-master.png`, and replaces both `dist/Luma Swarm Studio.app` and its clean signed ZIP. The command-line release binary and packaged application therefore come from the same build.

## Creator and support

- Support: [Buy Me a Coffee](https://buymeacoffee.com/CHCOfficial)
- Code: [GitHub · CHCOfficial](https://github.com/CHCOfficial)
- Graphics: [DeviantArt · CHCOfficial](https://www.deviantart.com/chcofficial)
- Audio: [Suno · artfulexpchc](https://suno.com/@artfulexpchc)

## Licence

The source code may be used, modified, and redistributed subject to the attribution and retained-link conditions in [`LICENSE.md`](LICENSE.md). Graphics and audio remain copyright CHCOfficial and are not placed under the code permission; the licence grants only the limited project-distribution rights stated there. This is a custom licence, not MIT.

## Architecture

| Module | Responsibility |
|---|---|
| `app` | Native editor shell, transport, formation browser, inspector, timeline, presentation mode |
| `renderer` + `shaders/` | WGPU resources, GPU compute, instanced HDR scene, bloom/tone mapping |
| `simulation` | Fixed timestep, trajectory tracking, wind, hard separation/ground constraints, telemetry audit, tilt, rotors, battery |
| `trajectory` | Smooth deterministic transition curves |
| `assignment` | Hungarian and scalable spatial slot assignment |
| `formation` | Procedural and animated normalized RGBW point clouds |
| `formation_import` | OBJ surface/point-cloud sampling boundary |
| `image_formation` | Hole-free full-frame PNG/JPEG sampling, GIF decoding/playback, colour conversion, and RGBW slot generation |
| `safety_log` | Cumulative run incident accounting, representative diagnostics, and replay records |
| `timeline` | Launch, hold, morph, animation, colour, camera, and landing cues |
| `camera` | Showcase/orbit/free-fly rig and smooth framing |
| `project` | In-memory default show and imported-image state |
| `profiling` | Smoothed and worst-frame performance statistics |

## Tests

```bash
cargo test
```

Tests cover exact assignment uniqueness/cost, deterministic formation generation, free-fly movement and mouse look, trajectory endpoints and endpoint velocity, dense transition synchronisation with zero collision/ground incidents, hard overlap correction, a 5,000-aircraft solver-regression stress case, source-slot separation across every 17,000-drone showcase and Bonus formation, full-frame image preservation, hole-free raster coverage, RGBW reconstruction, GIF frame decoding/playback, cumulative incident episode accounting, OBJ sampling/normalization, and state serialization round trips.

## Extending

To add a formation, add a `FormationKind` variant and generator returning exactly the requested number of normalized `FormationPoint`s. Animation must be a deterministic function of `time` and `phase`. Add it to `FormationKind::SHOWCASE` to expose it in the browser.

To add a choreography operation, extend `CueKind`, its formation resolver, and `FleetSimulation::update_targets`.

For custom mesh formats, decode triangles into `Vec<Vec3>`, normalize to a radius near 10 metres, then reuse the deterministic barycentric surface sampling strategy in `formation_import`.
