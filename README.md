CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

## Part 1 - Core Path Tracer

The core renderer is a CUDA-based Monte Carlo path tracer supporting cosine-weighted diffuse scattering, multi-bounce indirect illumination, emissive surfaces, stream compaction, material sorting, and stochastic sampled antialiasing.

The primary ray is produced for each pixel starting from the camera. Based on the material and the outcome of the ray-scene intersection, the light ray terminates or continues to scatter. The above procedure repeats itself until the ray intersects with an emissive surface, exits the scene, or achieves its maximum depth of bounces.

---

### Diffuse BSDF and Multi-Bounce Path Tracing

For diffuse surfaces, I implemented cosine-weighted hemisphere sampling using the provided `calculateRandomDirectionInHemisphere()` function.

At every diffuse intersection, the current path throughput is multiplied by the material color:

```cpp
pathSegment.color *= material.color;
```

The ray origin is moved to the current surface intersection and a new direction is randomly sampled in the hemisphere around the surface normal.

Conceptually, each path follows:

```text
Camera Ray
    ↓
Intersection
    ↓
Diffuse BSDF Sampling
    ↓
New Ray
    ↓
Intersection
    ↓
...
```

When a path reaches an emissive material, its accumulated throughput is multiplied by the light color and emittance:

```cpp
path.color *= material.color * material.emittance;
```

The path then terminates. If a ray leaves the scene without reaching a light source, its contribution is set to zero. The renderer therefore supports indirect illumination through multiple diffuse bounces. This produces effects such as the red and green color bleeding visible in the Cornell box.

#### Early Multi-Bounce Result

The image below shows an early working result shortly after multi-bounce diffuse path tracing was enabled. At this point, the image is still dominated by Monte Carlo noise, but indirect illumination and color bleeding are already visible.

<p align="center">
  <img src="own_img/early_multibounce_noisy_cornell.png" width="70%">
</p>

---

### Stream Compaction

Many rays that bounce around the scene have paths that end before they get to the trace depth limit.

A path terminates when:

- it reaches an emissive surface,
- it escapes the scene, or
- it reaches the maximum allowed bounce depth.

Processing terminated rays is wasteful since they cannot contribute to any image generation. Therefore, stream compaction is done after each bounce.

The rendering loop follows the structure below:

```text
Ray-Scene Intersection
        ↓
      Shading
        ↓
Gather Terminated Paths
        ↓
  Stream Compaction
        ↓
 Continue Active Paths
```

The contribution of the final color from each terminated path to the image buffer is made by adding it to its respective `pixelIndex`.

The terminated paths are deleted from the list of active paths. The total count of active paths is obtained from the new end of the path buffer:

```cpp
num_paths = static_cast<int>(dev_path_end - dev_paths);
```

The rendering loop continues until no active paths remain.

#### Active Rays per Bounce

The usefulness of stream compaction increases with the number of bounces since the probability that the rays will end increases with increasing path depth.

The following data shows the number of active rays remaining after each bounce.

| Bounce | Active Rays |
| -----: | ----------: |
|      0 |        TODO |
|      1 |        TODO |
|      2 |        TODO |
|      3 |        TODO |
|      4 |        TODO |
|      5 |        TODO |
|      6 |        TODO |
|      7 |        TODO |
|      8 |        TODO |

<p align="center">
  <img src="img/stream_compaction_active_rays.png" width="75%">
</p>

As the number of alive rays becomes fewer, fewer rays would be processed by the subsequent intersection and shading kernels. In the absence of stream compaction, rays which have been terminated will still occupy work on the GPU.

#### Open Scene vs. Closed Scene

The success of stream compaction is based on the nature of the scene itself.

 An open scene is one in which rays leave the scene and end prematurely. Such rays are discarded straightaway from the active path buffer.

A closed scene will not allow any ray to escape the scene. Thus, a large number of paths are still alive for more bounces.

<table>
<tr>
<td align="center" width="50%">
<b>Open Scene</b><br><br>
<img src="img/stream_compaction_open_scene.png" width="100%">
</td>
<td align="center" width="50%">
<b>Closed Scene</b><br><br>
<img src="img/stream_compaction_closed_scene.png" width="100%">
</td>
</tr>
</table>

The difference in active-ray behavior between the two scenes is shown below.

<p align="center">
  <img src="img/stream_compaction_open_vs_closed.png" width="75%">
</p>

#### Stream Compaction Performance

All performance measurements were collected using the Release build with the same resolution, maximum trace depth, and scene configuration for each comparison.

| Scene  | Stream Compaction | Time / Frame |  FPS |
| ------ | ----------------- | -----------: | ---: |
| Open   | OFF               |      TODO ms | TODO |
| Open   | ON                |      TODO ms | TODO |
| Closed | OFF               |      TODO ms | TODO |
| Closed | ON                |      TODO ms | TODO |

#### Analysis

In the open scene, rays can escape early, causing the number of active paths to decrease quickly. Stream compaction prevents later intersection and shading kernels from processing a large number of terminated rays.

In the closed scene, rays are more likely to remain active until later bounce depths. Because fewer paths terminate early, stream compaction has less opportunity to reduce the workload during the first several bounces.

The overall performance benefit depends on whether the reduced intersection and shading workload outweighs the cost of performing stream compaction itself.

**TODO:** Replace or expand this paragraph using the final measured results.

---

### Material Sorting

Different material types may require different BSDF calculations. When neighboring GPU threads evaluate different material branches, warp divergence can reduce shading efficiency.

To improve shading coherence, I implemented material-based path sorting before the shading stage.

After ray-scene intersection, each active path receives a sorting key corresponding to the `materialId` of its intersection.

The rendering pipeline becomes:

```text
Ray-Scene Intersection
        ↓
 Build Material Keys
        ↓
 Sort by Material ID
        ↓
      Shading
        ↓
  Stream Compaction
```

The material IDs are used as sorting keys while the corresponding `PathSegment` and `ShadeableIntersection` data remain paired during sorting.

After sorting, paths interacting with the same material are contiguous in memory before shading.

Material sorting can be toggled on or off to directly compare its effect on performance.

#### Material Sorting Comparison

Both images below were rendered using the same Cornell box scene, resolution, trace depth, Release build configuration, and iteration count.

<table>
<tr>
<td align="center" width="50%">
<b>Material Sorting OFF</b><br><br>
<img src="img/material_sorting_off_release.png" width="100%">
</td>
<td align="center" width="50%">
<b>Material Sorting ON</b><br><br>
<img src="img/material_sorting_on_release.png" width="100%">
</td>
</tr>
</table>

#### Performance

| Configuration        | Time / Frame |  FPS |
| -------------------- | -----------: | ---: |
| Material Sorting OFF |      TODO ms | TODO |
| Material Sorting ON  |      TODO ms | TODO |

The relative performance change was:

**TODO% faster/slower with material sorting enabled.**

#### Analysis

Material sorting introduces additional GPU work because material keys must first be generated and the active paths must then be reordered.

However, grouping paths by material increases the probability that neighboring threads execute the same BSDF code path. This can reduce warp divergence during the shading stage.

In the Cornell box test, enabling material sorting changed the average frame time from **TODO ms/frame** to **TODO ms/frame**.

**TODO: Keep one of the following paragraphs depending on the final result.**

If material sorting improves performance:

> The reduction in shading divergence was large enough to outweigh the additional cost of generating material keys and sorting the active paths.

If material sorting decreases performance:

> The sorting overhead was larger than the shading benefit in this scene. Most non-emissive materials currently use the same diffuse BSDF, so the amount of material-dependent branch divergence is relatively small.

The potential benefit of material sorting should become larger as the renderer supports more computationally different BSDFs, such as diffuse reflection, specular reflection, and refraction.

---

### Stochastic Sampled Antialiasing

Without stochastic antialiasing, every iteration generates the primary camera ray using the same fixed location inside each pixel.

Repeatedly sampling the same sub-pixel location can produce visible aliasing along object boundaries.

I implemented stochastic sampled antialiasing by generating two independent random offsets for every pixel and iteration:

```cpp
float jitterX = u01(rng);
float jitterY = u01(rng);

float sampleX = (float)x + jitterX;
float sampleY = (float)y + jitterY;
```

The jittered coordinates are then used to generate the primary ray direction.

Instead of repeatedly sampling one fixed location inside a pixel:

```text
+---------+
|         |
|    X    |
|         |
+---------+
```

different iterations sample different sub-pixel positions:

```text
+---------+
|  •      |
|      •  |
|    •    |
| •     • |
+---------+
```

As the number of iterations increases, these samples are averaged together, producing smoother estimates of pixel coverage along geometry boundaries.

#### Antialiasing Comparison

Both images below were rendered using the same scene, resolution, maximum trace depth, and iteration count.

<table>
<tr>
<td align="center" width="50%">
<b>Antialiasing OFF</b><br><br>
<img src="img/antialiasing_off_500.png" width="100%">
</td>
<td align="center" width="50%">
<b>Stochastic Antialiasing ON</b><br><br>
<img src="img/antialiasing_on_500.png" width="100%">
</td>
</tr>
</table>

The effect of stochastic antialiasing is most visible along high-contrast geometry boundaries, such as sphere silhouettes, light-source edges, and diagonal edges.

At low iteration counts, Monte Carlo path-tracing noise can obscure the antialiasing improvement. As the image converges, stochastic sub-pixel sampling produces smoother boundaries than repeatedly sampling the same position within every pixel.

**TODO:** Replace this sentence with the final visual observation from the AA comparison.

---

### Part 1 Summary

The completed core renderer includes:

- CUDA-based Monte Carlo path tracing
- Cosine-weighted diffuse BSDF sampling
- Multi-bounce indirect illumination
- Emissive surface lighting
- Path termination for escaped rays and exhausted bounce depth
- Stream compaction of terminated paths
- Active-ray analysis across bounce depth
- Open-scene and closed-scene stream compaction comparison
- Toggleable material sorting
- Material sorting performance comparison
- Stochastic sub-pixel antialiasing

These features form the base renderer used for the additional rendering and performance features implemented in Part 2.
