import React from 'react';
import car_image from '../../assets/carHomePage.jpg';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';
function HomePage() {
    const navigate = useNavigate();

    return (
        <div className="homepage">
            <div className="image-container">
                <img src={car_image} alt="Background" className="background-image" />
                <div className="gradient-overlay">
                    <div className="text-overlay">
                        <h1>Welcome to CARALYZE</h1>
                        <p className="tagline">Unlock the True Value of Your Car in a Click.</p>
                    </div>
                </div>
            </div>
            <div className="overlay">
                <h2>Car Price Prediction of</h2>
                <div className="options">
                    <div onClick={() => navigate('/prediction/old-car')} className="option">
                        Old Car
                    </div>
                    <div onClick={() => navigate('/prediction/new-car')} className="option">
                        New Car
                    </div>
                    <div onClick={() => navigate('/prediction/manual-entry')} className="option">
                        Manual Enter Details of Car
                    </div>
                </div>
            </div>
        </div>
    );
}

export default HomePage;