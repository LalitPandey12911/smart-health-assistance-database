const form = document.querySelector('.health-form');
if (form) {
    form.addEventListener('submit', function (e) {
        const symptomInput = document.getElementById("symptoms");
        const errorMessage = document.getElementById("symptom-error");
        const symptomsArray = symptomInput.value.split(",").map(s => s.trim()).filter(Boolean);

        if (symptomsArray.length < 3) {
            e.preventDefault();
            errorMessage.style.display = "block";
            return;
        } else {
            errorMessage.style.display = "none";
        }

        const loader = document.getElementById('loader');
        loader.style.display = 'flex';
    });
}

document.body.classList.add('page-fade');

function toggleLoader(isLoading) {
    const loader = document.getElementById('loader');
    loader.style.display = isLoading ? 'flex' : 'none';
}

window.addEventListener('load', () => {
    toggleLoader(false);
});
