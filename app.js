document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startButton');

    startButton.addEventListener('click', async () => {
        try {
            
            var BoardIds  = require("brainflow").BoardIds;
            var BoardShim = require("brainflow").BoardShim;
            function sleep (ms)
            {
                return new Promise ((resolve) => { setTimeout (resolve, ms); });
            }
            // Initialize BrainFlow
            const board = new BoardShim(-1, {});

            // Start streaming
            board.startStream();
            console.log("Streaming started");
            await sleep(3000);
            board.stopStream();
            const data = board.getBoardData();
            board.releaseSession();
            console.info('Data');
            console.info(data);
        } catch (error) {
            console.error("Error during streaming:", error);
        }
    });
});

