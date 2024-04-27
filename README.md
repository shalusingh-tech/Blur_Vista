# BlurVista👓

Automated web application for selectively blurring objects within uploaded videos based on user-defined parameters.

## Overview

Blur Vista is a web application designed to provide users with an intuitive platform to upload videos and automatically  blur specific objects within those videos. This project offers a solution for privacy concerns or content editing where certain objects need to be obscured. Users can define the objects they want to blur, set the time range for the blurring effect, and adjust the blur intensity according to their preferences.

## Features

- **Upload Video:** Users can easily upload videos directly through the web interface.
- **Selective Object Blur:** Users have the ability to select specific objects within the video that they want to blur.
- **Time Range Selection:** Users can define the time range during which the selected objects will be blurred.
- **Adjustable Blur Intensity:** The intensity of the blur effect can be adjusted to suit the user's requirements.
- **Automatic Blurring:** Once the parameters are set, the application automatically applies the blur effect to the specified objects within the defined time range.

## How to Use

1. **Upload Video:** Begin by uploading the video file you want to edit.
2. **Select Objects:** Identify the objects within the video that you wish to blur. These could be faces, license plates, sensitive information, etc.
3. **Set Time Range:** Specify the time range during which the selected objects should be blurred. This could be a specific duration or throughout the entire video.
4. **Adjust Blur Intensity:** Customize the intensity of the blur effect according to your preference.
5. **Apply Blur:** Once all parameters are set, initiate the blur process to automatically apply the effect to the specified objects within the defined time range.
6. **Preview and Save:** Preview the edited video to ensure the desired blurring effect is achieved. Save the edited video once satisfied with the results.


## Technologies Used

- **SDK:** Gradio
- **Machine Learning:** TensorFlow
- **Image Processing:** OpenCV, PIL
- **Data Manipulation:** NumPy


## Run Locally

Clone the project

```bash
  git clone https://github.com/shalusingh-tech/Blur_Vista.git
```

Go to the project directory

```bash
  cd Blur_Vista
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

## User Interface

![img](https://github.com/shalusingh-tech/Blur_Vista/assets/68162265/08971973-fce1-47fe-9921-304040f6bac1)
![img](https://github.com/shalusingh-tech/Blur_Vista/assets/68162265/687f8922-1e5e-4260-8805-3bf9ab8b7453)



