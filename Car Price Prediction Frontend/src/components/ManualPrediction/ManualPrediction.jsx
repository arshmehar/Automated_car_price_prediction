import React, { useState } from 'react'
import axios from 'axios';
import './ManualPrediction.css'

const ManualPrediction = () => {
    const [formData, setFormData] = useState({
        manufacturer: '', model: '', production_year: '',
        category: '', leather_interior: '', fuel_type: '', engine_volume: '',
        mileage: '', cylinders: '', gear_box_type: '', drive_wheels: '',
        doors: '', wheels: '', airbags: '',
    });
    const [predictedPrice, setPredictedPrice] = useState(null);

    const manufacturers = [
        'LEXUS', 'HONDA', 'FORD', 'HYUNDAI', 'MERCEDES-BENZ', 'CHEVROLET',
        'TOYOTA', 'BMW', 'NISSAN', 'AUDI', 'VOLKSWAGEN', 'KIA', 'MAZDA'
    ];

    const categories = [
        'Jeep', 'Hatchback', 'Coupe', 'Sedan', 'SUV', 'Convertible', 'Pickup',
        'Van', 'Wagon', 'Crossover'
    ];

    const fuelTypes = ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Electric'];

    const gearBoxTypes = [
        'Automatic', 'Variator', 'Manual', 'Tiptronic', 'Dual-clutch'
    ];

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const conversionRate = 0.012;
        const priceInUSD = parseFloat(formData.price) * conversionRate;
        const levyInUSD = parseFloat(formData.levy) * conversionRate;

        const dataToSend = {
            ...formData,
            price: priceInUSD,
            levy: levyInUSD
        };

        // try {
        //     const response = await axios.post('http://localhost:4000/predict', dataToSend);
        //     setPredictedPrice(response.data.predicted_price);
        // } catch (error) {
        //     console.error('Error making prediction:', error);
        // }
    };

    return (
        <div className='app-container'>
            <h1>Car Price Prediction</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Manufacturer:</label>
                    <select
                        name="manufacturer"
                        value={formData.manufacturer}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select Manufacturer</option>
                        {manufacturers.map((manufacturer, index) => (
                            <option key={index} value={manufacturer}>{manufacturer}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label>Category:</label>
                    <select
                        name="category"
                        value={formData.category}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select Category</option>
                        {categories.map((category, index) => (
                            <option key={index} value={category}>{category}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label>Leather Interior:</label>
                    <div>
                        <label>
                            <input
                                type="radio"
                                name="leather_interior"
                                value="Yes"
                                checked={formData.leather_interior === 'Yes'}
                                onChange={handleChange}
                                required
                            />
                            Yes
                        </label>
                        <label>
                            <input
                                type="radio"
                                name="leather_interior"
                                value="No"
                                checked={formData.leather_interior === 'No'}
                                onChange={handleChange}
                                required
                            />
                            No
                        </label>
                    </div>
                </div>

                <div>
                    <label>Fuel Type:</label>
                    <select
                        name="fuel_type"
                        value={formData.fuel_type}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select Fuel Type</option>
                        {fuelTypes.map((fuel, index) => (
                            <option key={index} value={fuel}>{fuel}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label>Gear Box Type:</label>
                    <select
                        name="gear_box_type"
                        value={formData.gear_box_type}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select Gear Box Type</option>
                        {gearBoxTypes.map((gearBox, index) => (
                            <option key={index} value={gearBox}>{gearBox}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label>Airbags:</label>
                    <input
                        type="number"
                        name="airbags"
                        value={formData.airbags}
                        onChange={handleChange}
                        placeholder="Enter Number of Airbags"
                        required
                    />
                </div>
                <div>
                    <label>Model:</label>
                    <input
                        type="text"
                        name="model"
                        value={formData.model}
                        onChange={handleChange}
                        placeholder="Enter car model"
                        required
                    />
                </div>
                <div>
                    <label>Production Year:</label>
                    <input
                        type="text"
                        name="production_year"
                        value={formData.production_year}
                        onChange={handleChange}
                        placeholder="Enter the production year"
                        required
                    />
                </div>
                <div>
                    <label>Engine Volume:</label>
                    <input
                        type="text"
                        name="engine_volume"
                        value={formData.engine_volume}
                        onChange={handleChange}
                        placeholder="Enter the engine volume in Liters"
                        required
                    />
                </div>
                <div>
                    <label>Mileage:</label>
                    <input
                        type="text"
                        name="mileage"
                        value={formData.mileage}
                        onChange={handleChange}
                        placeholder="Enter the mileage"
                        required
                    />
                </div>
                <div>
                    <label>Cylinders:</label>
                    <input
                        type="text"
                        name="cylinders"
                        value={formData.cylinders}
                        onChange={handleChange}
                        placeholder="How many cylinders are there in car?"
                        required
                    />
                </div>
                <div>
                    <label>Doors:</label>
                    <input
                        type="text"
                        name="doors"
                        value={formData.doors}
                        onChange={handleChange}
                        placeholder="Enter the number of doors the car has"
                        required
                    />
                </div>

                <button type="submit" className='predict-price-btn'>Predict Price</button>
            </form>

            {predictedPrice !== null && (
                <h2>Predicted Price: {predictedPrice}</h2>
            )}
        </div>
    );
}

export default ManualPrediction
