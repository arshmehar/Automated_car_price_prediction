import React, { useState, useEffect } from 'react';
import './Navbar.css';
import { useNavigate } from 'react-router-dom'; 
import { useAuthContext } from '../../context/AuthContext';
import profileIcon from '../../assets/profile_icon.png'; 

const Navbar = () => {
  const { isAuthenticated, logout } = useAuthContext(); 
  const navigate = useNavigate(); 
  const [showLogout, setShowLogout] = useState(false); 

  const handleLogout = () => {
    logout(); 
    navigate('/'); 
  };

  const toggleLogoutMenu = () => {
    setShowLogout(!showLogout);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      const menu = document.querySelector('.logout-menu');
      const icon = document.querySelector('.profile-icon');
      if (icon && !icon.contains(event.target) && menu && !menu.contains(event.target)) {
        setShowLogout(false); 
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <nav className="navbar">
      <div className="navbar__brand">Caralyze</div>
      <div className="navbar__auth">
        {isAuthenticated ? (
          <div className="user-profile">
            <div
              className="profile-icon"
              onClick={toggleLogoutMenu}
              style={{ cursor: 'pointer' }}
            >
              <img src={profileIcon} alt="Profile" />
            </div>
            {showLogout && (
              <div className="logout-menu">
                <button onClick={handleLogout}>Logout</button>
              </div>
            )}
          </div>
        ) : (
          <button onClick={() => navigate('/signup')} className="navbar__signup-btn">
            Sign Up
          </button>
        )}
      </div>
    </nav>
  );
};

export default Navbar;