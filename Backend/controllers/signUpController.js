const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const User = require('../models/UserSchema');

// Signup Route
const signup = async (req, res) => {
    const { name, email, username, password } = req.body;

    // Regular expression to validate the username
    // special symbols allowed in userName are - {@,$,!,%,*,?,&,#}
    const usernameRegex = /^(?=.*[A-Z])(?=.*\d)(?=.*[@$#!%*?&])[A-Za-z\d@$#!%*?&]{6,}$/;

    try {
        // Check if username is unique
        const existingUsername = await User.findOne({ username });
        if (existingUsername) {
            return res.status(400).json({ message: 'Username already exists' });
        }

        // Validate username format
        if (!usernameRegex.test(username)) {
            return res.status(400).json({ 
                message: 'Username must contain at least one uppercase letter, one special symbol, one number, and be at least 6 characters long' 
            });
        }

        // Check if email is unique
        const existingEmail = await User.findOne({ email });
        if (existingEmail) {
            return res.status(400).json({ message: 'Email already exists' });
        }

        // Hash the password before saving
        const hashedPassword = await bcrypt.hash(password, 12);

        // Create a new user
        const newUser = new User({ name, email, username, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ message: 'User created successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Something went wrong' });
    }
};

// Login Route
const signin = async (req, res) => {
    //user can login either with username or email
    const { identifier, password } = req.body; // identifier can be either username or email

    // Check for missing fields
    if (!identifier || !password) {
        return res.status(400).json({ message: 'identifier (username or email) and password are required' });
    }

    try {
        // Determine if the identifier is an email or username
        const isEmail = /\S+@\S+\.\S+/.test(identifier);
        const user = await User.findOne(isEmail ? { email: identifier } : { username: identifier });

        if (!user) {
            return res.status(401).json({ message: 'User not found' });
        }

        // Check if the password matches
        const isPasswordCorrect = await bcrypt.compare(password, user.password);
        if (!isPasswordCorrect) {
            return res.status(401).json({ message: 'Invalid password' });
        }

        // Generate JWT token
        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });

        // Respond with token and user data
        res.json({ message: 'Logged in successfully!', token });
    } catch (error) {
        res.status(500).json({ message: 'An error occurred during login', error });
    }
};

module.exports = {signup, signin};