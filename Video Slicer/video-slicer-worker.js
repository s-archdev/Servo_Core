// Worker script for handling video processing in the background
self.onmessage = function(e) {
  const { action, data } = e.data;
  
  if (action === 'processFrames') {
    // Simulate processing time
    const { totalFrames } = data;
    let processed = 0;
    
    // Report progress periodically
    const interval = setInterval(() => {
      processed += 1;
      const progress = Math.min(Math.round((processed / totalFrames) * 100), 99);
      
      self.postMessage({
        type: 'progress',
        progress: progress
      });
      
      // When reaching 100%, complete the process
      if (processed >= totalFrames) {
        clearInterval(interval);
        self.postMessage({
          type: 'complete',
          message: 'Processing complete!'
        });
      }
    }, 100);
  }
};
