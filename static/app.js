const formatter = new Intl.NumberFormat("en-US");

const estimateForm = document.querySelector("#estimate-form");
const estimateButton = document.querySelector("#estimate-button");
const cantonSelect = document.querySelector("#canton");
const zipCodeSelect = document.querySelector("#zip-code");
const propertyTypeSelect = document.querySelector("#property-type");
const resultPanel = document.querySelector("#result-panel");
const monthlyRent = document.querySelector("#monthly-rent");
const annualRent = document.querySelector("#annual-rent");
const errorMessage = document.querySelector("#error-message");
const technicalDetails = document.querySelector("#technical-details");
const detailZipCode = document.querySelector("#detail-zip-code");
const detailTaxIndex = document.querySelector("#detail-tax-index");
const detailDistance = document.querySelector("#detail-distance");
const detailModelVersion = document.querySelector("#detail-model-version");
const detailModelType = document.querySelector("#detail-model-type");

let zipCodesByCanton = {};

function createOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function replaceOptions(selectElement, options, selectedValue) {
  selectElement.replaceChildren();

  options.forEach((optionValue) => {
    const option = createOption(optionValue, optionValue);
    selectElement.appendChild(option);
  });

  if (selectedValue && options.includes(selectedValue)) {
    selectElement.value = selectedValue;
  }
}

function updateZipCodeOptions() {
  const selectedCanton = cantonSelect.value;
  const zipCodes = zipCodesByCanton[selectedCanton] || [];
  replaceOptions(zipCodeSelect, zipCodes.map(String), zipCodes[0] ? String(zipCodes[0]) : "");
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

async function loadOptions() {
  const response = await fetch("/api/options");
  if (!response.ok) {
    throw new Error("Could not load property options.");
  }

  const optionsPayload = await response.json();
  zipCodesByCanton = optionsPayload.zipCodesByCanton;

  replaceOptions(cantonSelect, optionsPayload.cantons, optionsPayload.defaultCanton);
  replaceOptions(propertyTypeSelect, optionsPayload.subtypes, optionsPayload.defaultSubtype);
  updateZipCodeOptions();
}

function buildPredictionRequest() {
  return {
    area: Number(document.querySelector("#area").value),
    rooms: Number(document.querySelector("#rooms").value),
    floor: Number(document.querySelector("#floor").value),
    canton: cantonSelect.value,
    zipCode: Number(zipCodeSelect.value),
    propertyType: propertyTypeSelect.value,
    hasLake: document.querySelector("#has-lake").checked,
    isNew: document.querySelector("#is-new").checked,
    isQuiet: document.querySelector("#is-quiet").checked,
  };
}

function renderPrediction(predictionPayload) {
  monthlyRent.textContent = `CHF ${formatter.format(predictionPayload.monthlyRent)} / month`;
  annualRent.textContent = `CHF ${formatter.format(predictionPayload.annualRent)} / year`;
  resultPanel.hidden = false;

  detailZipCode.textContent = predictionPayload.details.zipCode;
  detailTaxIndex.textContent = predictionPayload.details.taxIndex;
  detailDistance.textContent = `${predictionPayload.details.distanceToNearestHubKm} km`;
  detailModelVersion.textContent = predictionPayload.details.modelVersion;
  detailModelType.textContent = predictionPayload.details.modelType;
  technicalDetails.hidden = false;
}

async function requestPrediction(event) {
  event.preventDefault();
  clearError();
  estimateButton.disabled = true;
  estimateButton.textContent = "Estimating...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildPredictionRequest()),
    });
    const responsePayload = await response.json();

    if (!response.ok) {
      throw new Error(responsePayload.error || "Prediction failed.");
    }

    renderPrediction(responsePayload);
  } catch (error) {
    showError(error.message);
  } finally {
    estimateButton.disabled = false;
    estimateButton.textContent = "Estimate rent";
  }
}

cantonSelect.addEventListener("change", updateZipCodeOptions);
estimateForm.addEventListener("submit", requestPrediction);

loadOptions().catch((error) => {
  estimateButton.disabled = true;
  showError(error.message);
});
