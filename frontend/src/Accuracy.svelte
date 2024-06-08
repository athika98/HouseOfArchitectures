<script>
    import { onMount } from "svelte";
    import Chart from "chart.js/auto";
    import { currentPage } from "./store";

    let valAccuracyChart;
    let valLossChart;

    let valAccuracyData = [];
    let valLossData = [];

    const fetchResults = async () => {
        const response = await fetch("http://localhost:5000/results/results.json");
        const data = await response.json();
        valAccuracyData = data.map(result => ({
            model: result.model,
            valAccuracy: result.val_accuracy[result.val_accuracy.length - 1] // Last epoch's val_accuracy
        }));
        valLossData = data.map(result => ({
            model: result.model,
            valLoss: result.val_loss[result.val_loss.length - 1] // Last epoch's val_loss
        }));
        createCharts();
    };

    const createCharts = () => {
        const ctx1 = document.getElementById('valAccuracyChart').getContext('2d');
        const ctx2 = document.getElementById('valLossChart').getContext('2d');

        if (valAccuracyChart) valAccuracyChart.destroy();
        if (valLossChart) valLossChart.destroy();

        valAccuracyChart = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: valAccuracyData.map(d => d.model),
                datasets: [{
                    label: 'Validation Accuracy',
                    data: valAccuracyData.map(d => d.valAccuracy),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        valLossChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: valLossData.map(d => d.model),
                datasets: [{
                    label: 'Validation Loss',
                    data: valLossData.map(d => d.valLoss),
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    };

    onMount(() => {
        fetchResults();
    });
</script>

<div>
    <img
        src="image2.jpg"
        style="width: 100%; margin-bottom: 20px; border-radius: 5px;"
        alt="Architecture"
    />
    <h3>✨ Accuracy and Loss Comparison ✨</h3>
    <p>Here you can see the different models and their Accuracy and Loss.</p>
</div>

<div>
    <h2>Validation Accuracy</h2>
    <canvas id="valAccuracyChart"></canvas>
</div>

<div>
    <h2>Validation Loss</h2>
    <canvas id="valLossChart"></canvas>
</div>

<style>
    :global(body) {
        margin: 0;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 100vh;
        background-color: #f8f8f8;
    }

    nav {
        background-color: #e8d9cd;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
        gap: 1em;
        width: 100%;
        max-width: 600px;
    }

    nav button {
        background-color: transparent;
        border: none;
        color: white;
        padding: 0.5em 1em;
        border-radius: 4px;
        transition: background-color 0.3s;
        cursor: pointer;
        flex: 1;
        text-align: center;
    }

    nav button:hover {
        background-color: #e8d9cd;
        color: black;
    }

    div {
        text-align: center;
        margin: 20px auto;
        max-width: 600px;
    }

    canvas {
        width: 100% !important;
        max-width: 600px;
        height: 400px;
        margin: 20px auto;
    }
</style>
