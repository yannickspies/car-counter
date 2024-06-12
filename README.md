# Installation

## Clone the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/yannickspies/car-counter
cd car-counter
```

## Set Up Virtual Environment

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Install Dependencies

```
pip install -r requirements.txt
```

# Usage

This script processes video files to detect cars and buses, drawing bounding boxes around detected objects and displaying the number of detected objects on each frame.

## Running the Script

Place your input videos in the input directory. Then run the script:

```
python detect.py
```
