import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import dotenv from "dotenv";
import colors from 'colors';
import cors from 'cors';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const httpServer = createServer(app);
const io = new Server(httpServer, {
    cors: {
        origin: "*", // Adjust according to your frontend's URL for production
        methods: ["GET", "POST"],
    },
});

// Use middleware
app.use(cors()); // Enable CORS
app.use(express.json()); // To parse JSON bodies

let dataStorage = {};

app.post('/send-data', (req, res) => {
    dataStorage = req.body;
    console.log("Received data:", dataStorage);
    io.emit('dataUpdated', dataStorage); // Emit an update event to all clients
    return res.status(200).json({ message: "Data received successfully" });
});

app.get('/get-data', (req, res) => {
    return res.status(200).json(dataStorage);
});

// Handle WebSocket connections
io.on('connection', (socket) => {
    console.log('A client connected'.green);

    socket.on('disconnect', () => {
        console.log('A client disconnected'.red);
    });
});

httpServer.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`.yellow.bold);
});
