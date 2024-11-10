import React, { useState } from 'react';
import axios from 'axios';
import './NewCarPrediction.css'
import newCar from '../../assets/newCar.jpg'

const NewCarPrediction = () => {
    const [manufacturer, setManufacturer] = useState('');
    const [model, setModel] = useState('');
    const [carData, setCarData] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async () => {
        setLoading(true);
        const currentYear = new Date().getFullYear();
        const minYear = currentYear - 10;
        let url = 'https://api.api-ninjas.com/v1/cars?limit=15';
        if (manufacturer) url += `&make=${manufacturer}`;
        if (model) url += `&model=${model}`;

        try {
            const response = await axios.get(url, {
                headers: {
                    'X-Api-Key': '0KY3k31ZsNVpos4FT7fCQw==Fi5IN15bhrbSL7jy',
                },
            });
            const filteredData = response.data.filter(car => car.year >= minYear && car.year <= currentYear);
            setCarData(filteredData);
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
        <div className="new-container">
            <div className="new-car-page">
                <div className="new-form-container">
                    <h2>Search for New Car</h2>
                    <form className="new-car-form">
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
                    <img src={newCar} alt="Car" className="car-image" />
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

export default NewCarPrediction
