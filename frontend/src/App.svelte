<script>
    import Accuracy from "./Accuracy.svelte";
    import { currentPage } from "./store";

    let selectedModel = "resnet";
    let models = ["custom_cnn_model", "archinet", "resnet", "inceptionv3"];
    let file = null;
    let prediction = null;
    let imageUrl = null;

    const handleSubmit = async () => {
        if (file) {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("model", selectedModel);

            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                prediction = data.predictions;
            } else {
                console.error("Error:", response.statusText);
            }
        } else {
            alert("Please select a file");
        }
    };

    const handleFileChange = (event) => {
        file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                imageUrl = reader.result;
            };
            reader.readAsDataURL(file);
        }
    };

    const navigateTo = (page) => {
        currentPage.set(page);
    };
</script>

<main>
    <h1>House of Architectures</h1>
    <p>Architecture is just art we live in.</p>

    <nav>
        <button on:click={() => navigateTo("home")}>Classification</button>
        <button on:click={() => navigateTo("accuracy")}>Accuracy Comparison</button>
    </nav>

    {#if $currentPage === "home"}
        <div>
            <img
                src="image2.jpg"
                style="width: 100%; margin-bottom: 20px; border-radius: 5px;"
                alt="Architecture"
            />
            <h3>✨ Image Classification ✨</h3>
            <p>
                How it works: Select a model, upload a picture of a building, and
                click on submit.
            </p>
            <div class="form-group-horizontal">
                <label for="model">Select Model:</label>
                <select bind:value={selectedModel} id="model">
                    {#each models as model}
                        <option value={model}>{model}</option>
                    {/each}
                </select>
            </div>

            <div class="form-group-horizontal">
                <label for="file-upload">Upload Image:</label>
                <input type="file" accept="image/*" id="file-upload" on:change={handleFileChange} />
            </div>

            {#if imageUrl}
                <!-- svelte-ignore a11y-img-redundant-alt -->
                <img id="uploaded-image" src={imageUrl} alt="Uploaded Image" />
            {/if}

            <div class="form-group">
                <button on:click={handleSubmit}>Submit</button>
            </div>

            {#if prediction}
                <div class="prediction-container">
                    <p><strong>Prediction:</strong></p>
                    <span>{JSON.stringify(prediction)}</span> <!-- Display prediction in a single line -->
                </div>
            {/if}
        </div>
    {:else}
        <Accuracy />
    {/if}
</main>

<style>
    :global(body) {
        margin: 0;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 100vh;
        background-color: #f8f8f8;
        overflow-x: hidden;
    }

    main {
        background-color: #fff;
        text-align: center;
        padding: 1em;
        max-width: 600px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        width: 90%;
    }

    nav {
        background-color: #e8d9cd;
        padding: 1em;
        border-radius: 8px;
        display: flex;
        justify-content: space-around;
        gap: 1em;
        width: 100%;
        max-width: 570px;
        margin: 0 auto 20px auto;
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

    .form-group-horizontal {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 1em;
        margin-bottom: 1em;
        width: 100%;
    }

    .form-group-horizontal label {
        font-weight: bold;
    }

    input[type="file"], select {
        padding: 0.5em;
        border: 1px solid #ccc;
        border-radius: 4px;
        width: 100%;
    }

    button {
        padding: 0.5em 1em;
        background-color: #e8d9cd;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        width: 100%;
    }

    button:hover {
        background-color: #e8d9cd;
        color: black;
    }

    .prediction-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 0.5em;
        text-align: left;
        background: #e8d9cd98;
        padding: 1em;
        border-radius: 5px;
        width: 100%;
        max-width: 570px;
    }

    #uploaded-image {
        max-width: 100%;
        margin-top: 20px;
        border-radius: 5px;
        width: 200px;
    }

    @media (min-width: 768px) {
        .form-group-horizontal {
            flex-direction: row;
        }

        button {
            width: auto;
        }
    }
</style>
