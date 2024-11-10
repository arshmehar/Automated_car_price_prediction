import React, { useState } from 'react';
import axios from 'axios';
import './OldCarPrediction.css';
import oldCar from '../../assets/oldCar.jpg';

const OldCarPrediction = () => {
    const [manufacturer, setManufacturer] = useState('');
    const [model, setModel] = useState('');
    const [carData, setCarData] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async () => {
        setLoading(true);
        const currentYear = new Date().getFullYear();
        const thresholdYear = currentYear - 10;
        let url = 'https://api.api-ninjas.com/v1/cars?limit=15';
        if (manufacturer) url += `&make=${manufacturer}`;
        if (model) url += `&model=${model}`;

        try {
            const response = await axios.get(url, {
                headers: {
                    'X-Api-Key': '0KY3k31ZsNVpos4FT7fCQw==Fi5IN15bhrbSL7jy',
                },
            });
            // Filter the data to include only cars older than the threshold year
            const oldCars = response.data.filter(car => car.year < thresholdYear);
            setCarData(oldCars);
        } catch (error) {
            console.error('Error fetching car data:', error);
        } finally {
            setLoading(false);
        }
    };

    const handlePredictPrice = (car) => {
        // Add logic here for price prediction, e.g., make an API call or perform calculations
        console.log(`Predicting price for: ${car.make} ${car.model} (${car.year})`);
        // For example:
        alert(`Price prediction feature is not implemented yet for ${car.make} ${car.model}.`);
    };

    return (
        <div className="old-container">
            <div className="old-car-page">
                <div className="old-form-container">
                    <h2>Search for Old Car</h2>
                    <form className="old-car-form">
                        <div className="form-group">
                            <label htmlFor="manufacturer">Manufacturer</label>
                            <input
                                type="text"
                                id="manufacturer"
                                value={manufacturer}
                                onChange={(e) => setManufacturer(e.target.value)}
                                placeholder="Enter manufacturer"
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="model">Model</label>
                            <input
                                type="text"
                                id="model"
                                value={model}
                                onChange={(e) => setModel(e.target.value)}
                                placeholder="Enter model"
                            />
                        </div>
                        <button type="button" onClick={handleSearch} className="search-button">
                            Search
                        </button>
                    </form>
                </div>
                <div className="image-container">
                    <img src={oldCar} alt="Car" className="car-image" />
                </div>
            </div>

            {loading && <div>Loading...</div>}

            <div className="car-cards-container">
                {carData.length > 0 ? (
                    carData.map((car, index) => (
                        <div key={index} className="car-card">
                            <h3>{car.make} {car.model}</h3>
                            <p><strong>Year:</strong> {car.year}</p>
                            <p><strong>Cylinders:</strong> {car.cylinders}</p>
                            <p><strong>Fuel Type:</strong> {car.fuel_type}</p>
                            <p><strong>Transmission:</strong> {car.transmission}</p>    
                            {/* tells manual or automatic */}
                            <button 
                                onClick={() => handlePredictPrice(car)} 
                                className="predict-price-button"
                            >
                                Predict Price
                            </button>
                        </div>
                    ))
                ) : (
                    <div>No cars found.</div>
                )}
            </div>
        </div>
    );
}

export default OldCarPrediction;