---
sidebar_position: 4
---

# Unity for High-Fidelity Rendering

## Chapter Objectives

By the end of this chapter, you will be able to:
- Set up Unity for robotics simulation with ROS integration
- Create photorealistic environments for Physical AI training
- Implement realistic lighting and material systems
- Connect Unity to ROS 2 using ROS# or similar bridges
- Understand the role of high-fidelity rendering in Physical AI

## Unity in Physical AI Context

Unity has emerged as a powerful platform for creating high-fidelity simulation environments for Physical AI applications. Unlike Gazebo which focuses on physics accuracy, Unity excels in:

### Visual Realism
- Photorealistic rendering capabilities
- Advanced lighting and material systems
- Realistic environmental effects
- High-quality visual assets

### Machine Learning Applications
- Synthetic data generation for computer vision
- Domain randomization for robust perception
- Photorealistic training environments
- Visual-inertial simulation

## Setting Up Unity for Robotics

### Prerequisites
- Unity Hub and Unity Editor (2021.3 LTS or newer recommended)
- Visual Studio or similar IDE for C# development
- Basic understanding of Unity concepts (scenes, GameObjects, components)

### Installing ROS# for Unity
ROS# is a popular Unity package for ROS communication:

1. Download the ROS# Unity package from the Unity Asset Store or GitHub
2. Import it into your Unity project
3. Configure network settings for ROS communication

### Basic Unity-ROS Setup

```csharp
// Example: Basic ROS connection script
using UnityEngine;
using RosSharp.RosBridgeClient;

public class UnityRosConnector : MonoBehaviour
{
    public RosSocket rosSocket;
    public string rosBridgeServerUrl = "ws://localhost:9090";

    void Start()
    {
        // Connect to ROS bridge
        rosSocket = new RosSocket(new WebSocketNetSharpProtocol(rosBridgeServerUrl));

        Debug.Log("Connected to ROS bridge: " + rosBridgeServerUrl);
    }

    void OnDestroy()
    {
        rosSocket.Close();
    }
}
```

## Creating Photorealistic Environments

### Environment Design Principles

For Physical AI applications, environments should be:
- **Physically Accurate**: Realistic physics and material properties
- **Visually Diverse**: Varied lighting conditions and textures
- **Functionally Relevant**: Match real-world deployment scenarios
- **Computationally Efficient**: Balanced realism with performance

### Lighting Systems

Unity offers multiple lighting approaches for Physical AI:

```csharp
// Example: Dynamic lighting system for day/night cycles
using UnityEngine;
using System.Collections;

public class DynamicLightingSystem : MonoBehaviour
{
    public Light sunLight;
    public float dayDuration = 120f; // 2 minutes for full day/night cycle
    private float timeOfDay = 0.5f; // 0 = midnight, 0.5 = noon, 1 = midnight

    void Update()
    {
        // Update time of day
        timeOfDay += Time.deltaTime / dayDuration;
        if (timeOfDay >= 1) timeOfDay = 0;

        // Update sun position and intensity
        float sunAngle = timeOfDay * 360f - 90f; // Start at sunrise
        transform.rotation = Quaternion.Euler(sunAngle, 0, 0);

        // Adjust intensity based on sun angle
        float intensity = Mathf.Clamp01(Mathf.Cos(Mathf.Deg2Rad * sunAngle));
        sunLight.intensity = Mathf.Lerp(0.1f, 1.0f, intensity);
    }
}
```

### Material Systems for Physical AI

Creating realistic materials for sensor simulation:

```csharp
// Example: Material property randomization for domain adaptation
using UnityEngine;

public class MaterialRandomizer : MonoBehaviour
{
    public Material[] materials;
    public float roughnessRange = 0.5f;
    public float metallicRange = 0.3f;

    void Start()
    {
        RandomizeMaterials();
    }

    public void RandomizeMaterials()
    {
        foreach (Renderer renderer in GetComponentsInChildren<Renderer>())
        {
            Material mat = renderer.material;

            // Randomize material properties
            mat.SetFloat("_Roughness", Random.Range(0.1f, roughnessRange));
            mat.SetFloat("_Metallic", Random.Range(0.0f, metallicRange));

            // Add texture variations
            if (mat.HasProperty("_Color"))
            {
                Color baseColor = mat.GetColor("_Color");
                Color randomColor = baseColor * Random.ColorHSV(0.8f, 1.2f, 0.8f, 1.2f, 0.8f, 1.2f);
                mat.SetColor("_Color", randomColor);
            }
        }
    }
}
```

## Sensor Simulation in Unity

### Camera Systems

Creating realistic camera systems for computer vision training:

```csharp
// Example: Configurable camera with noise simulation
using UnityEngine;
using System.Collections;

public class PhysicalAICamera : MonoBehaviour
{
    [Header("Camera Settings")]
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    public float fieldOfView = 60f;

    [Header("Noise Settings")]
    public bool enableNoise = true;
    public float noiseIntensity = 0.01f;
    public float blurIntensity = 0.1f;

    private Camera cam;
    private RenderTexture renderTexture;

    void Start()
    {
        cam = GetComponent<Camera>();
        SetupCamera();
    }

    void SetupCamera()
    {
        // Configure camera properties
        cam.fieldOfView = fieldOfView;

        // Create render texture for processing
        renderTexture = new RenderTexture(resolutionWidth, resolutionHeight, 24);
        cam.targetTexture = renderTexture;
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (enableNoise)
        {
            // Apply noise and distortion effects
            ApplyCameraEffects(source, destination);
        }
        else
        {
            Graphics.Blit(source, destination);
        }
    }

    void ApplyCameraEffects(RenderTexture source, RenderTexture destination)
    {
        // Apply noise, blur, or other camera effects here
        // This would typically use a custom shader
        Graphics.Blit(source, destination);
    }
}
```

### LiDAR Simulation in Unity

Unity can simulate LiDAR sensors using raycasting:

```csharp
// Example: Unity-based LiDAR simulation
using UnityEngine;
using System.Collections.Generic;

public class UnityLidar : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public int horizontalSamples = 360;
    public int verticalSamples = 16;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float minRange = 0.1f;
    public float maxRange = 10.0f;
    public LayerMask detectionMask;

    [Header("Output Settings")]
    public bool publishToRos = true;

    private List<float> ranges;
    private float[] angles;

    void Start()
    {
        ranges = new List<float>(horizontalSamples);
        angles = new float[horizontalSamples];

        // Precompute angles
        float angleStep = (maxAngle - minAngle) / horizontalSamples;
        for (int i = 0; i < horizontalSamples; i++)
        {
            angles[i] = minAngle + i * angleStep;
        }
    }

    void Update()
    {
        if (publishToRos)
        {
            ScanEnvironment();
        }
    }

    void ScanEnvironment()
    {
        ranges.Clear();

        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = angles[i];
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionMask))
            {
                ranges.Add(hit.distance);
            }
            else
            {
                ranges.Add(maxRange);
            }
        }

        // Publish ranges to ROS if connected
        PublishScanData();
    }

    void PublishScanData()
    {
        // Convert to ROS LaserScan message format and publish
        // Implementation depends on ROS# or other bridge used
    }
}
```

## Unity-ROS Integration

### ROS# Bridge Configuration

Setting up the ROS# bridge for Unity:

```csharp
// Example: ROS publisher for sensor data
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor;

public class UnityCameraPublisher : MonoBehaviour
{
    private RosSocket rosSocket;
    private string imageTopic = "/unity_camera/image_raw";
    private string cameraInfoTopic = "/unity_camera/camera_info";

    public Camera unityCamera;
    private Texture2D capturedImage;

    void Start()
    {
        // Connect to ROS bridge
        rosSocket = new RosSocket(new WebSocketNetSharpProtocol("ws://localhost:9090"));

        // Initialize camera
        SetupCamera();
    }

    void SetupCamera()
    {
        // Configure camera resolution and settings
        RenderTexture rt = new RenderTexture(640, 480, 24);
        unityCamera.targetTexture = rt;
        capturedImage = new Texture2D(640, 480, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (Time.frameCount % 30 == 0) // Publish every 30 frames (approx 2 Hz if 60 FPS)
        {
            CaptureAndPublishImage();
        }
    }

    void CaptureAndPublishImage()
    {
        // Capture the camera output
        RenderTexture.active = unityCamera.targetTexture;
        capturedImage.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
        capturedImage.Apply();

        // Convert to ROS image message
        ImageMessage imageMsg = CreateImageMessage(capturedImage);

        // Publish to ROS
        rosSocket.Publish(imageTopic, imageMsg);
    }

    ImageMessage CreateImageMessage(Texture2D texture)
    {
        ImageMessage msg = new ImageMessage();
        msg.header = new Messages.Standard.Header();
        msg.header.stamp = new Time();
        msg.header.frame_id = "unity_camera_optical_frame";

        msg.height = (uint)texture.height;
        msg.width = (uint)texture.width;
        msg.encoding = "rgb8";
        msg.is_bigendian = 0;
        msg.step = (uint)(texture.width * 3); // 3 bytes per pixel for RGB

        // Convert texture to byte array
        byte[] imageData = texture.EncodeToJPG();
        msg.data = imageData;

        return msg;
    }
}
```

## Domain Randomization for Physical AI

### Randomization Techniques

Domain randomization helps create robust perception systems:

```csharp
// Example: Environment randomization system
using UnityEngine;
using System.Collections.Generic;

public class DomainRandomizer : MonoBehaviour
{
    [Header("Lighting Randomization")]
    public Light[] lights;
    public Color[] lightColors;
    public float[] lightIntensities;

    [Header("Material Randomization")]
    public List<Renderer> randomizableRenderers;
    public Material[] randomMaterials;

    [Header("Object Placement")]
    public List<GameObject> randomObjects;
    public Bounds randomizationBounds;

    [Header("Timing")]
    public float randomizationInterval = 10f;
    private float lastRandomizationTime;

    void Start()
    {
        lastRandomizationTime = Time.time;
        RandomizeEnvironment();
    }

    void Update()
    {
        if (Time.time - lastRandomizationTime > randomizationInterval)
        {
            RandomizeEnvironment();
            lastRandomizationTime = Time.time;
        }
    }

    public void RandomizeEnvironment()
    {
        // Randomize lighting
        foreach (Light light in lights)
        {
            light.color = lightColors[Random.Range(0, lightColors.Count)];
            light.intensity = lightIntensities[Random.Range(0, lightIntensities.Length)];
        }

        // Randomize materials
        foreach (Renderer renderer in randomizableRenderers)
        {
            Material randomMat = randomMaterials[Random.Range(0, randomMaterials.Length)];
            renderer.material = randomMat;
        }

        // Randomize object positions
        foreach (GameObject obj in randomObjects)
        {
            Vector3 randomPos = new Vector3(
                Random.Range(randomizationBounds.min.x, randomizationBounds.max.x),
                Random.Range(randomizationBounds.min.y, randomizationBounds.max.y),
                Random.Range(randomizationBounds.min.z, randomizationBounds.max.z)
            );
            obj.transform.position = randomPos;
        }
    }
}
```

## Physical AI Training Applications

### Synthetic Data Generation

Unity can generate large datasets for training:

```csharp
// Example: Data generation system
using UnityEngine;
using System.IO;

public class DataGenerator : MonoBehaviour
{
    public Camera dataCamera;
    public string datasetPath = "Assets/Datasets/";
    public int samplesPerSession = 1000;
    private int sampleCounter = 0;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            CaptureTrainingData();
        }
    }

    void CaptureTrainingData()
    {
        // Capture RGB image
        Texture2D rgbImage = CaptureCameraImage(dataCamera);

        // Generate ground truth (if needed)
        Texture2D depthImage = GenerateDepthMap();
        Texture2D segmentationMask = GenerateSegmentationMask();

        // Save images with consistent naming
        string baseName = Path.Combine(datasetPath, $"sample_{sampleCounter:0000}");

        File.WriteAllBytes($"{baseName}_rgb.png", rgbImage.EncodeToPNG());
        File.WriteAllBytes($"{baseName}_depth.png", depthImage.EncodeToPNG());
        File.WriteAllBytes($"{baseName}_mask.png", segmentationMask.EncodeToPNG());

        sampleCounter++;
        Debug.Log($"Captured sample {sampleCounter}");
    }

    Texture2D CaptureCameraImage(Camera cam)
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        image.Apply();

        RenderTexture.active = currentRT;
        return image;
    }

    Texture2D GenerateDepthMap()
    {
        // Implementation depends on your specific needs
        // Could use Unity's depth texture or raycasting
        return new Texture2D(640, 480);
    }

    Texture2D GenerateSegmentationMask()
    {
        // Generate semantic segmentation mask
        return new Texture2D(640, 480);
    }
}
```

## Performance Optimization

### Balancing Realism and Performance

For Physical AI applications in Unity:

1. **Level of Detail (LOD)**: Use LOD groups for complex objects
2. **Occlusion Culling**: Hide objects not visible to sensors
3. **Texture Streaming**: Load textures on demand
4. **Light Baking**: Precompute static lighting

### Multi-Threaded Processing

```csharp
// Example: Asynchronous image processing
using System.Threading.Tasks;
using UnityEngine;

public class AsyncImageProcessor : MonoBehaviour
{
    public async void ProcessImageAsync(Texture2D image)
    {
        await Task.Run(() => {
            // Perform heavy image processing in background thread
            ProcessImageOnBackgroundThread(image);
        });

        // Return to main thread for Unity operations
        await Task.Delay(1); // Yield to Unity's main thread
    }

    void ProcessImageOnBackgroundThread(Texture2D image)
    {
        // Heavy computation here (e.g., neural network inference)
        // Don't call Unity APIs from this thread!
    }
}
```

## Best Practices for Physical AI in Unity

### Environment Design
- Create diverse environments matching deployment scenarios
- Include realistic lighting variations
- Add environmental effects (weather, time of day)
- Use domain randomization for robustness

### Sensor Accuracy
- Match Unity sensor parameters to real hardware
- Include realistic noise and distortion models
- Validate simulation against real sensor data
- Consider computational constraints

### Integration Considerations
- Maintain consistent coordinate frames between Unity and ROS
- Ensure proper timing synchronization
- Implement error handling for network interruptions
- Plan for scalability of simulation environments

## Chapter Summary

Unity provides high-fidelity rendering capabilities essential for Physical AI applications requiring photorealistic simulation. The platform excels in generating synthetic training data, implementing domain randomization, and creating visually realistic environments. When integrated with ROS through bridges like ROS#, Unity becomes a powerful tool for Physical AI development, complementing physics-focused simulators like Gazebo.

## Exercises

1. Create a Unity scene with a robot model and configure a camera with realistic parameters.
2. Implement a simple domain randomization system that changes lighting and materials.
3. Set up a basic Unity-ROS connection and publish camera images to ROS topics.

## Next Steps

In the next chapter, we'll explore building test environments in both Gazebo and Unity for Physical AI applications, including obstacle courses, navigation challenges, and manipulation scenarios.