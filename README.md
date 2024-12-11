
# Automated Vehicle Valuation using Machine learning

This project predicts the price of a car based on various features (like manufacturer, model, mileage, etc.) using a machine learning model. The backend of the project uses Python, Flask, and various ML libraries, and it communicates with a frontend built using Node.js and Express.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Python 3.x**: Required for running the backend Python services.
- **Node.js**: Required for the frontend Node.js and Express part of the application.
- **pip**: Python package installer.
- **git**: To clone the repository and manage version control.

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/jaskirank1/Car-Price-Prediction.git
cd Car-Price-Prediction
```

### 2. Setup Virtual Environment for Python

The project uses a **virtual environment** for isolating Python dependencies. This ensures that dependencies for the backend (such as ML libraries) don't interfere with the global Python environment.

#### Steps to create and activate a virtual environment:

1. **Create a virtual environment:**

   On **Windows**:

   ```bash
   python -m venv venv
   ```

   On **Mac/Linux**:

   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**

   On **Windows** (PowerShell):

   ```bash
   .\venv\Scripts\Activate
   ```

   On **Mac/Linux** (Terminal):

   ```bash
   source venv/bin/activate
   ```

3. **Install Python dependencies**:

   Once the virtual environment is activated, install the necessary Python packages by running:

   ```bash
   pip install -r backend/requirements.txt
   ```

   This will install all the required libraries such as Flask, Numpy, Scikit-learn, etc.

### 3. Install Node.js Dependencies

1. **Install Node.js dependencies**:

   Navigate to the **frontend** directory and install the required packages for the frontend (Node.js):

   ```bash
   cd frontend
   npm install
   ```

### 4. Running the Project

#### Running the Backend (Python Flask):

1. **Activate the virtual environment** (if not already activated):

   Navigate to the **backend/models** directory 
   ```bash
   cd backend/models
   source venv/bin/activate  # or .\venv\Scripts\Activate for Windows
   ```

2. **Run the Flask server**:

   Navigate to the **backend** directory and run the Flask server:

   ```bash
   cd backend
   python model_server.py
   ```

   This will start the backend server that serves the car price prediction model.

#### Running the Frontend (Node.js):

1. **Start the frontend**:

   Go back to the **frontend** directory and run:

   ```bash
   cd frontend
   npm run dev
   ```

   This will launch the frontend application (React) and connect to the backend for car price predictions.

### 5. Deactivating the Virtual Environment

Once you're done with the Python part of the project, you can deactivate the virtual environment:

```bash
deactivate
```

This will return you to your global environment.

## Working with the Project

- **Frontend Development**: If you are working on the frontend (React + Node.js), you don’t need to activate the virtual environment.
- **Backend Development**: If you are working on the backend (Python + Flask), make sure the virtual environment is activated.

## Common Issues

- **Missing Python Dependencies**: If you see errors related to missing Python packages, make sure the virtual environment is activated and that you have installed all dependencies with `pip install -r backend/requirements.txt`.
- **Port Conflicts**: The Flask app and React app may both run on different ports by default (e.g., 5000 for Flask and 3000 for React). Ensure these ports are not in conflict with other running services.
