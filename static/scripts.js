const form = document.querySelector('.health-form');
if (form) {
    form.addEventListener('submit', function (e) {
        const symptomInput = document.getElementById("symptoms");
        const errorMessage = document.getElementById("symptom-error");
        const symptomsArray = symptomInput.value.split(",").map(s => s.trim()).filter(Boolean);

        if (symptomsArray.length < 3) {
            e.preventDefault();
            if (errorMessage) {
                errorMessage.style.display = "block";
                errorMessage.style.color = "#d9363e";
                errorMessage.style.fontWeight = "500";
                errorMessage.style.marginTop = "10px";
            }
            return;
        } else {
            if (errorMessage) {
                errorMessage.style.display = "none";
            }
        }

        const loader = document.getElementById('loader');
        if (loader) {
            loader.style.display = 'flex';
        }
    });
}

function goBack() {
    window.history.back();
}

function toggleLoader(isLoading) {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.display = isLoading ? 'flex' : 'none';
    }
}

document.body.classList.add('page-fade');

window.addEventListener('load', () => {
    toggleLoader(false);

    const flashMessages = document.querySelectorAll('.flash');
    if (flashMessages.length > 0) {
        setTimeout(() => {
            flashMessages.forEach(msg => {
                msg.style.transition = "opacity 0.5s ease, transform 0.5s ease";
                msg.style.opacity = "0";
                msg.style.transform = "translateY(-10px)";
                setTimeout(() => msg.remove(), 500);
            });
        }, 5000);
    }
});