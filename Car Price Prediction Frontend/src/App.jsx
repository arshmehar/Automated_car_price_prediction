import React from 'react';
import './App.css'
import { Route, Routes } from 'react-router-dom';
import ManualPrediction from './components/ManualPrediction/ManualPrediction';
import Navbar from './components/Navbar/Navbar';
import Signup from './components/Signup/Signup';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import HomePage from './components/HomePage/HomePage';
import OldCarPrediction from './components/OldCarPrediction/OldCarPrediction';
import NewCarPrediction from './components/NewCarPrediction/NewCarPrediction';

function App() {
    return (
        <div>
            <ToastContainer/>
            <Navbar/>
            <Routes>
                <Route path='/' element={<HomePage/>}/>
                <Route path='/signup' element={<Signup />} />
                <Route path='/prediction/old-car' element={<OldCarPrediction/>} />
                <Route path='/prediction/new-car' element={<NewCarPrediction/>} />
                <Route path='/prediction/manual-entry' element={<ManualPrediction/>} />
            </Routes>
        </div>
    );
}

export default App;