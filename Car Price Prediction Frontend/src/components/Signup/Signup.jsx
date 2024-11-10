import React, { useState } from 'react';
import './Signup.css';
import axios from 'axios';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useNavigate } from 'react-router-dom';
import { useAuthContext } from '../../context/AuthContext';

const Signup = () => {
    const { login } = useAuthContext();
    const [isSignup, setIsSignup] = useState(true);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        username: '',
        password: ''
    });
    const navigate = useNavigate();

    const toggleForm = () => {
        setIsSignup(!isSignup);
        setFormData({ name: '', email: '', username: '', password: '' });
    };

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (isSignup) {
                const response = await axios.post('http://localhost:4000/api/authUser/signup', formData);
                toast.success(response.data.message);
            } else {
                const identifier = formData.email || formData.username;
                if (!identifier) {
                    toast.error('Please provide either email or username');
                    return;
                }
                const { password } = formData;
                const response = await axios.post('http://localhost:4000/api/authUser/signin', { 
                    identifier, password 
                });

                const { token } = response.data;
                
                // Use the context's login function to update the state
                login(token);

                navigate('/');
                toast.success('Logged in successfully!');
            }
        } catch (error) {
            toast.error(error.response?.data?.message || 'Something went wrong');
        }
    };

    return (
        <div className="auth-card">
            <h2>{isSignup ? 'Sign Up' : 'Sign In'}</h2>
            <form onSubmit={handleSubmit}>
                {isSignup && (
                    <>
                        <input type="text" name="name" placeholder="Name" value={formData.name} onChange={handleChange} required />
                        <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} required />
                        <input type="text" name="username" placeholder="Username" value={formData.username} onChange={handleChange} required />
                    </>
                )}

                {/* Email and Username fields for Login with "OR" between them */}
                {!isSignup && (
                    <>
                        <div className="email-username-container">
                            <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} />
                            <span className="or-text">OR</span>
                            <input type="text" name="username" placeholder="Username" value={formData.username} onChange={handleChange} />
                        </div>
                    </>
                )}

                <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} required />
                <button type="submit">{isSignup ? 'Create Account' : 'Login'}</button>
            </form>
            <p onClick={toggleForm} className="toggle-link">
                {isSignup ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
            </p>
        </div>
    );
};

export default Signup;