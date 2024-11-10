const express = require('express');
const axios = require('axios');
const cors = require('cors');
const connectDB = require('./config/db');
const signupRoute = require('./routes/signupRoute');
require('dotenv').config();

const app = express();
connectDB();
app.use(cors());
app.use(express.json());

// Route to handle user signups
app.use('/api/authUser', signupRoute);

// Prediction endpoint that communicates with Python backend
app.post('/predict', async (req, res) => {
    try {
        // Send request to the Python server
        const response = await axios.post('http://localhost:5000/predict', req.body);
        res.json(response.data);  // Respond with the prediction result
    } catch (error) {
        console.error('Error calling Python API:', error);
        res.status(500).send('Error making prediction');
    }
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});