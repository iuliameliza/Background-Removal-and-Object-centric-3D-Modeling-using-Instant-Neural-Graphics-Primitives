# Background Removal and Object-centric 3D Modeling using Instant Neural Graphics Primitives

## Description

The tool incorporates five prominent background removal techniques: `rembg`,  `backgroundremoval`, `OpenCV`, `scikit-image`, and `Pillow`, providing users with a variety of  options to meet their specific requirements. These methods apply machine learning, color  segmentation, iterative refinement, and image manipulation to accurately eliminate the  background and isolate the foreground. In addition, the tool includes methods such as  determining the optimal frames per second and video resizing to maximize the system's  performance and efficiency.

## Repository Address

blabla

## Build Steps

1. Clone the repository locally: 

   ```git clone https://github.com/iuliameliza/Martuica_Iulia-Eliza_License.git```

2. Using a Python-enabled terminal, run the following command: 
   
   ```pip install -r requirements.txt```

## Launch Steps

### View one of the provided examples:

1. Using a Python-enabled terminal, run the following command:

    ```instant-ngp.exe "path/to/example"```

2. After the GUI is opened, wait for the 3D object to render.
   
3. To generate the mesh of the object, open the second panel called `instant-ngp v1.0dev`.

4. Look for the panel called `Export mesh / volume / slices`. 
   
5. Click the button `Mesh it!` to generate the mesh.

6. Click the button `Save it!` to save the '.obj' file in the corresponding folder. 

### Creating a new 3D Object

1. Record a video while rotating slowly around an object such that after 1 minute you reach the initial point.

2. Place the recording inside a new folder located at `/data/nerf/new_folder`

3. Using a Python-enabled terminal, run the following command: 
   
   `python path/to/project/scripts/colmap2nerf.py --run_colmap --aabb_scale 1 --bgrmv 0 --video_in video.mp4`

4. In addition, if the video resolution exceeds that of High Definition (HD), it is recommended to include the optional parameter `--resize` in the command.

5. The value used for the `--aabb_scale` should be either 1 or 2, depending on the object's dimensions and its frame occupancy.

6. To visualize the generated object, follow the steps previously mentioned.