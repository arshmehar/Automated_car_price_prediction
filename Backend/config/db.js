const mongoose = require("mongoose");

const connectDB = async()=>{
    try{
        await mongoose.connect('mongodb+srv://kiran18202:xLpuzFcosFkynodB@user.osvzj.mongodb.net/carPricePrediction');
        console.log("MongoDB Connected");
    }
    catch (err) {
        console.log("MongoDB Connection Error ", err);
    }
}

module.exports = connectDB;