document.getElementById("recommendForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const product = document.getElementById("product").value;

    // Call the backend API
    const response = await fetch(`/recommend?product_id=${product}`);
    const data = await response.json();

    const recommendationList = document.getElementById("recommendationList");
    recommendationList.innerHTML = ""; // Clear old recommendations

    // Populate new recommendations
    data.recommendations.forEach((recommendation) => {
        const listItem = document.createElement("li");
        listItem.textContent = recommendation;
        recommendationList.appendChild(listItem);
    });
});
