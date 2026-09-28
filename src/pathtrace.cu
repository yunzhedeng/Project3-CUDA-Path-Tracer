#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>

#define ERRORCHECK 0

#define STREAM_COMPACTION 1

#define SORT_BY_MATERIAL 0

#define ANTI_ALIASING 0

#define RUSSIAN_ROULETTE 0
#define RR_START_DEPTH 3

#define DEPTH_OF_FIELD 0
#define DOF_APERTURE_RADIUS 0.70f
#define DOF_FOCAL_DISTANCE 10.5f

#define REFRACTION 0

#define DIRECT_LIGHTING 0

#define LOW_DISCREPANCY_SAMPLING 0

#define MOTION_BLUR 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_material_ids = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));
    cudaMemset(dev_material_ids, 0, pixelcount * sizeof(int));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_material_ids);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + y * cam.resolution.x;
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float jitterX = u01(rng);
        float jitterY = u01(rng);

        float sampleX = (float)x + jitterX;
        float sampleY = (float)y + jitterY;

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        glm::vec3 primaryDirection;

        if (ANTI_ALIASING == 0)
        {
            primaryDirection = glm::normalize( cam.view - cam.right * cam.pixelLength.x *
                ((float)x - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );
        }
        else
        {
            primaryDirection = glm::normalize( cam.view - cam.right * cam.pixelLength.x *
                (sampleX - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
            );
        }

        segment.ray.origin = cam.position;
        segment.ray.direction = primaryDirection;

        segment.ray.time = MOTION_BLUR ? u01(rng) : 0.0f;

        if (DEPTH_OF_FIELD == 1)
        {
            float lensU = u01(rng);
            float lensV = u01(rng);

            float radius = DOF_APERTURE_RADIUS * sqrtf(lensU);
            float angle = 6.28318530718f * lensV;

            glm::vec3 lensOffset = cam.right * (radius * cosf(angle)) + cam.up * (radius * sinf(angle));

            glm::vec3 viewDirection = glm::normalize(cam.view);

            float focusT = DOF_FOCAL_DISTANCE / glm::dot(primaryDirection, viewDirection);

            glm::vec3 focalPoint = cam.position + primaryDirection * focusT;

            segment.ray.origin = cam.position + lensOffset;

            segment.ray.direction = glm::normalize( focalPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        segment.allowEmission = true;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths && pathSegments[path_index].remainingBounces > 0){
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool hitOutside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            bool tmpOutside = true;
            
            Ray motionRay = pathSegment.ray;
            glm::vec3 motionOffset = glm::vec3(0.0f);

            if (MOTION_BLUR == 1)
            {
                motionOffset = geom.motion * pathSegment.ray.time;
                motionRay.origin -= motionOffset;
            }
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, motionRay, tmp_intersect, tmp_normal, tmpOutside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, motionRay, tmp_intersect, tmp_normal, tmpOutside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                hitOutside = tmpOutside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = hitOutside;
        }
    }
}

__global__ void buildMaterialKeys(
    int num_paths,
    ShadeableIntersection* intersections,
    int* material_keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        if (intersections[idx].t > 0.0f)
        {
            material_keys[idx] = intersections[idx].materialId;
        }
        else
        {
            material_keys[idx] = -1;
        }
    }
}

__device__ bool visibleToLight(
    glm::vec3 origin,
    glm::vec3 direction,
    float lightDistance,
    int lightIndex,
    Geom* geoms,
    int geoms_size)
{
    Ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = direction;

    for (int i = 0; i < geoms_size; i++)
    {
        if (i == lightIndex)
        {
            continue;
        }

        Geom& geom = geoms[i];

        glm::vec3 tmpIntersect;
        glm::vec3 tmpNormal;
        bool tmpOutside = true;

        float t = -1.0f;

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(
                geom,
                shadowRay,
                tmpIntersect,
                tmpNormal,
                tmpOutside
            );
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(
                geom,
                shadowRay,
                tmpIntersect,
                tmpNormal,
                tmpOutside
            );
        }

        if (t > 0.0f && t < lightDistance - 0.002f)
        {
            return false;
        }
    }

    return true;
}

__device__ glm::vec3 sampleDirectLighting(
    PathSegment& pathSegment,
    glm::vec3 intersectPoint,
    glm::vec3 surfaceNormal,
    const Material& surfaceMaterial,
    Geom* geoms,
    int geoms_size,
    Material* materials,
    bool useLowDiscrepancy,
    int iter,
    int depth,
    thrust::default_random_engine& rng)
{
    int lightIndex = -1;

    for (int i = 0; i < geoms_size; i++)
    {
        if (geoms[i].type == CUBE &&
            materials[geoms[i].materialid].emittance > 0.0f)
        {
            lightIndex = i;
            break;
        }
    }

    if (lightIndex == -1)
    {
        return glm::vec3(0.0f);
    }

    Geom light = geoms[lightIndex];
    Material lightMaterial = materials[light.materialid];

    float u;
    float v;

    if (useLowDiscrepancy)
    {
        unsigned int sampleIndex = (unsigned int)(iter + 1);
        glm::vec2 sample = halton2D(sampleIndex, 5, 7);

        unsigned int seed = (unsigned int)pathSegment.pixelIndex ^ ((unsigned int)depth * 0x9e3779b9u) ^ 0x85ebca6bu;

        sample.x += hashToUnitFloat(seed);
        sample.y += hashToUnitFloat(seed ^ 0xc2b2ae35u);

        sample.x -= floorf(sample.x);
        sample.y -= floorf(sample.y);

        u = sample.x - 0.5f;
        v = sample.y - 0.5f;
    }
    else
    {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        u = u01(rng) - 0.5f;
        v = u01(rng) - 0.5f;
    }

    glm::vec3 localLightPoint( u, -0.5f, v);

    glm::vec3 lightPoint = multiplyMV(light.transform, glm::vec4(localLightPoint, 1.0f));

    glm::vec3 lightNormal = glm::normalize( multiplyMV( light.invTranspose, glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));

    glm::vec3 toLight = lightPoint - intersectPoint;

    float distanceSquared = glm::dot(toLight, toLight);

    float distance = sqrtf(distanceSquared);

    glm::vec3 lightDirection = toLight / distance;

    float surfaceCos = fmaxf( glm::dot(surfaceNormal, lightDirection), 0.0f);

    float lightCos = fmaxf( glm::dot(lightNormal, -lightDirection), 0.0f);

    if (surfaceCos <= 0.0f ||
        lightCos <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 shadowOrigin =
        intersectPoint +
        surfaceNormal * 0.001f;

    if (!visibleToLight(shadowOrigin, lightDirection, distance, lightIndex, geoms, geoms_size))
    {
        return glm::vec3(0.0f);
    }

    float lightArea = fabsf(light.scale.x * light.scale.z);

    glm::vec3 emittedLight = lightMaterial.color * lightMaterial.emittance;

    glm::vec3 directContribution = pathSegment.color * surfaceMaterial.color * emittedLight *
        (
            surfaceCos *
            lightCos *
            lightArea /
            (PI * distanceSquared)
        );

    return directContribution;
}
// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                if (DIRECT_LIGHTING == 0 ||
                    pathSegments[idx].allowEmission)
                {
                    pathSegments[idx].color *=materialColor * material.emittance;
                }
                else
                {
                    pathSegments[idx].color = glm::vec3(0.0f);
                }

                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 intersectPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
                thrust::default_random_engine rng =
                    makeSeededRandomEngine(
                        iter,
                        pathSegments[idx].pixelIndex,
                        pathSegments[idx].remainingBounces
                    );

                if (DIRECT_LIGHTING == 1 &&
                    material.hasRefractive <= 0.0f)
                {
                    glm::vec3 directContribution =
                        sampleDirectLighting(
                            pathSegments[idx],
                            intersectPoint,
                            intersection.surfaceNormal,
                            material,
                            geoms,
                            geoms_size,
                            materials,
                            LOW_DISCREPANCY_SAMPLING == 1,
                            iter,
                            depth,
                            rng
                        );

                    image[pathSegments[idx].pixelIndex] += directContribution;
                }
                scatterRay(
                    pathSegments[idx],
                    intersectPoint,
                    intersection.surfaceNormal,
                    material,
                    intersection.outside,
                    REFRACTION == 1,
                    LOW_DISCREPANCY_SAMPLING == 1,
                    iter,
                    depth,
                    rng
                );
                pathSegments[idx].remainingBounces --;

                if (pathSegments[idx].remainingBounces == 0)
                {
                    pathSegments[idx].color = glm::vec3(0.0f);
                }
                else if (RUSSIAN_ROULETTE == 1 && depth >= RR_START_DEPTH)
                {
                    float survivalProbability = fmaxf(
                        pathSegments[idx].color.x,
                        fmaxf(
                            pathSegments[idx].color.y,
                            pathSegments[idx].color.z
                        )
                    );

                    survivalProbability = fminf(
                        0.95f,
                        fmaxf(0.05f, survivalProbability)
                    );

                    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
                    float rouletteSample = u01(rng);

                    if (rouletteSample > survivalProbability)
                    {
                        pathSegments[idx].color = glm::vec3(0.0f);
                        pathSegments[idx].remainingBounces = 0;
                    }
                    else
                    {
                        pathSegments[idx].color /= survivalProbability;
                    }
                }

            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces =0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct PathTerminated
{
    __host__ __device__
    bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* paths)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nPaths && paths[index].remainingBounces == 0)
    {
        image[paths[index].pixelIndex] += paths[index].color;
        paths[index].remainingBounces = -1;
    }
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    if (iter == 1)
    {
        std::cout << "Bounce 0: "
                << num_paths
                << " active rays"
                << std::endl;
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        if (SORT_BY_MATERIAL == 1){
            buildMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths,
                dev_intersections,
                dev_material_ids
            );
            checkCUDAError("build material keys");

            thrust::device_ptr<int> material_keys(dev_material_ids);

            auto zipped_begin = thrust::make_zip_iterator(
                thrust::make_tuple(dev_paths, dev_intersections)
            );

            auto zipped_end = zipped_begin + num_paths;

            thrust::sort_by_key(
                material_keys,
                material_keys + num_paths,
                zipped_begin
            );
        }
      
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_image
        );

        checkCUDAError("shade material");
        cudaDeviceSynchronize();

        gatherTerminatedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_image,
            dev_paths
        );

        checkCUDAError("gather terminated paths");
        cudaDeviceSynchronize();

        if (STREAM_COMPACTION == 1){

            dev_path_end = thrust::remove_if(
                thrust::device,
                dev_paths,
                dev_path_end,
                PathTerminated()
            );

            num_paths = static_cast<int>(dev_path_end - dev_paths);

            iterationComplete = (num_paths == 0);
        }
        else{
            iterationComplete = (depth >= traceDepth);
        }
                
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
