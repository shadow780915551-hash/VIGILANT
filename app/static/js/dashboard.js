
document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const videoFeed = document.getElementById('videoFeed');
    const cameraStatus = document.getElementById('cameraStatus');

    startBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/camera/start', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'camera started') {
                cameraStatus.textContent = 'Active';
                cameraStatus.style.color = '#10b981';
                videoFeed.src = '/camera/video_feed';
            }
        } catch (error) {
            console.error('Error starting camera:', error);
        }
    });

    stopBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/camera/stop', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'camera stopped') {
                cameraStatus.textContent = 'Stopped';
                cameraStatus.style.color = '#94a3b8';
                videoFeed.src = '';
            }
        } catch (error) {
            console.error('Error stopping camera:', error);
        }
    });

    setInterval(async () => {
        try {
            const response = await fetch('/api/alerts/recent');
            const data = await response.json();
            updateAlerts(data.alerts);
        } catch (error) {
            console.error('Error fetching alerts:', error);
        }
    }, 5000);
});

function updateAlerts(alerts) {
    const alertsList = document.getElementById('alertsList');
    if (alerts.length === 0) {
        alertsList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
        return;
    }

    alertsList.innerHTML = alerts.map(alert => `
        <div class="alert-item severity-${alert.severity.toLowerCase()}">
            <div class="alert-severity">${alert.severity}</div>
            <div class="alert-info">
                <div class="alert-confidence">Confidence: ${alert.confidence.toFixed(2)}</div>
                <div class="alert-detections">Detections: ${alert.num_detections}</div>
            </div>
        </div>
    `).join('');
}
