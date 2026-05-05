document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const btnText = startBtn.querySelector('.btn-text');
    const loader = startBtn.querySelector('.loader');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');
    const pcaStats = document.getElementById('pcaStats');
    const lbpStats = document.getElementById('lbpStats');
    const resultsPlot = document.getElementById('resultsPlot');
    const plotPlaceholder = document.getElementById('plotPlaceholder');

    startBtn.addEventListener('click', async () => {
        // UI Updates for loading
        startBtn.disabled = true;
        btnText.textContent = 'Processing...';
        loader.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsPlot.classList.add('hidden');
        plotPlaceholder.classList.remove('hidden');

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                displayResults(result.data);
            } else {
                throw new Error(result.message || 'An error occurred during evaluation');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to run evaluation: ' + error.message);
        } finally {
            // Restore UI state
            startBtn.disabled = false;
            btnText.textContent = 'Re-run Evaluation';
            loader.classList.add('hidden');
            loadingState.classList.add('hidden');
        }
    });

    function displayResults(data) {
        // Render PCA Stats
        pcaStats.innerHTML = generateStatsHTML(data.pca);
        
        // Render LBP Stats
        lbpStats.innerHTML = generateStatsHTML(data.lbp);

        // Render Plot
        // Add cache-busting query param so browser fetches new image if re-run
        resultsPlot.src = data.plot_url + '?t=' + new Date().getTime();
        
        resultsPlot.onload = () => {
            plotPlaceholder.classList.add('hidden');
            resultsPlot.classList.remove('hidden');
        };

        // Show Results container
        resultsContainer.classList.remove('hidden');
    }

    function generateStatsHTML(metrics) {
        const formatPercent = (val) => (val * 100).toFixed(2) + '%';
        const formatFloat = (val) => val.toFixed(4);

        return `
            <div class="stat-item">
                <span class="stat-label">Equal Error Rate (EER)</span>
                <span class="stat-value highlight">${formatPercent(metrics.eer)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Rank-1 Accuracy</span>
                <span class="stat-value">${formatPercent(metrics.rank1)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">D-Prime</span>
                <span class="stat-value">${formatFloat(metrics.d_prime)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">AUC</span>
                <span class="stat-value">${formatFloat(metrics.auc)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">TMR @ 1% FMR</span>
                <span class="stat-value">${formatPercent(metrics.tmr_1)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">TMR @ 0.01% FMR</span>
                <span class="stat-value">${formatPercent(metrics.tmr_001)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">FPIR @ EER</span>
                <span class="stat-value">${formatPercent(metrics.fpir)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">FNIR @ EER</span>
                <span class="stat-value">${formatPercent(metrics.fnir)}</span>
            </div>
        `;
    }
});
