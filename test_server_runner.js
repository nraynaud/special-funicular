// Combined test server and runner
// This file merges the functionality of test_server.js and run_test.js

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const { exec } = require('child_process');
const os = require('os');

// Configuration
const PORT = 3000;
const SERVER_URL = `http://localhost:${PORT}`;
const TEST_URL = `${SERVER_URL}/test.html`;

// Create a server
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // Handle static file requests
    if (req.method === 'GET' && (pathname === '/' || pathname === '/test.html')) {
        // Serve test.html
        const filePath = path.join(__dirname, 'test.html');
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading test.html');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    }
    // Handle JavaScript files
    else if (req.method === 'GET' && pathname.endsWith('.js')) {
        const filePath = path.join(__dirname, pathname);
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'application/javascript' });
            res.end(data);
        });
    }
    // Handle image files
    else if (req.method === 'GET' && (pathname.endsWith('.png') || pathname.endsWith('.jpg') || pathname.endsWith('.jpeg'))) {
        const filePath = path.join(__dirname, pathname);
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
                return;
            }
            const contentType = pathname.endsWith('.png') ? 'image/png' : 'image/jpeg';
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        });
    }
    // Handle test results submission and shutdown
    else if (req.method === 'POST' && pathname === '/save_results_and_shutdown') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const results = JSON.parse(body);
                const resultsPath = path.join(__dirname, 'test_results.json');
                fs.writeFile(resultsPath, JSON.stringify(results, null, 2), err => {
                    if (err) {
                        console.error('Error saving test results:', err);
                        res.writeHead(500);
                        res.end('Error saving test results');
                        return;
                    }
                    console.log('Test results saved to', resultsPath);
                    console.log('Shutting down server after saving results...');

                    res.writeHead(200);
                    res.end('Test results saved successfully. Server shutting down...');

                    // Give a short delay to allow the response to be sent
                    setTimeout(() => {
                        server.close(() => {
                            console.log('Server closed');
                            process.exit(0);
                        });
                    }, 100);
                });
            } catch (error) {
                console.error('Error parsing test results:', error);
                res.writeHead(400);
                res.end('Error parsing test results');
            }
        });
    }
    // Handle other requests
    else {
        res.writeHead(404);
        res.end('Not found');
    }
});

// This function is no longer needed as the browser will send results to /save_results_and_shutdown
// when tests complete, which will shut down the server automatically

// Start the server
console.log('Starting server...');
server.listen(PORT, () => {
    console.log(`Server running at ${SERVER_URL}/`);
    console.log(`Opening ${TEST_URL} in the default browser...`);

    // Determine the command to open the browser based on the operating system
    let command;
    switch (os.platform()) {
        case 'darwin': // macOS
            command = `open -gj "${TEST_URL}" --args  --no-startup-window`;
            break;
        case 'win32': // Windows
            command = `start "" "${TEST_URL}"`;
            break;
        default: // Linux and others
            command = `xdg-open "${TEST_URL}"`;
            break;
    }

    // Execute the command to open the browser
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error opening browser: ${error.message}`);
            server.close(() => {
                console.log('Server closed due to browser open error');
                process.exit(1);
            });
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            server.close(() => {
                console.log('Server closed due to browser stderr output');
                process.exit(1);
            });
            return;
        }
        console.log(`Browser opened successfully.`);

        // No need to set a timeout to check for test results
        // The browser will send results to /save_results_and_shutdown when tests complete
    });
});
