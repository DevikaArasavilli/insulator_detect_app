function showSection(id) {
  const sections = document.querySelectorAll('.section');
  sections.forEach(sec => sec.style.display = 'none');
  document.getElementById(id).style.display = 'block';
}

function toggleSection(id) {
  const section = document.getElementById(id);
  if (section.style.display === 'block') {
    section.style.display = 'none';
  } else {
    document.querySelectorAll('.toggle-section').forEach(sec => {
      sec.style.display = 'none';
    });
    section.style.display = 'block';
  }
}

window.onload = () => {
  document.querySelectorAll('.toggle-section').forEach(sec => {
    sec.style.display = 'none';
  });
};

document.addEventListener('DOMContentLoaded', function () {
  const fileInputs = {
    front: document.getElementById('frontInput'),
    back: document.getElementById('backInput'),
    left: document.getElementById('leftInput'),
    right: document.getElementById('rightInput')
  };

  const uploadBoxes = {
    front: document.getElementById('frontUpload'),
    back: document.getElementById('backUpload'),
    left: document.getElementById('leftUpload'),
    right: document.getElementById('rightUpload')
  };

  const previewContainer = document.getElementById('previewContainer');
  const uploadForm = document.getElementById('uploadForm');
  const resultsSection = document.getElementById('resultsSection');
  const resultsGrid = document.getElementById('resultsGrid');
  const statusCard = document.getElementById('statusCard');
  const spinner = document.getElementById('spinner');
  const combinedVisualization = document.getElementById('combinedVisualization');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');

  Object.keys(fileInputs).forEach(view => {
    const input = fileInputs[view];
    const box = uploadBoxes[view];

    function handleFile(file) {
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function (event) {
          box.style.borderColor = 'var(--primary)';
          box.style.backgroundColor = 'rgba(67, 97, 238, 0.05)';

          let previewBox = document.getElementById(`preview-${view}`);
          if (!previewBox) {
            previewContainer.style.display = 'grid';
            previewBox = document.createElement('div');
            previewBox.className = 'preview-box';
            previewBox.id = `preview-${view}`;
            previewBox.innerHTML = `
              <img src="${event.target.result}" alt="${view} view preview">
              <div class="preview-label">${view.charAt(0).toUpperCase() + view.slice(1)} View</div>
            `;
            previewContainer.appendChild(previewBox);
          } else {
            previewBox.querySelector('img').src = event.target.result;
          }
        };
        reader.readAsDataURL(file);
      }
    }

    input.addEventListener('change', function (e) {
      handleFile(e.target.files[0]);
    });

    box.addEventListener('dragover', e => {
      e.preventDefault();
      box.classList.add('drag-hover');
    });

    box.addEventListener('dragleave', e => {
      e.preventDefault();
      box.classList.remove('drag-hover');
    });

    box.addEventListener('drop', e => {
      e.preventDefault();
      box.classList.remove('drag-hover');
      const file = e.dataTransfer.files[0];
      if (file) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        handleFile(file);
      }
    });
  });

  uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();

    let allFilesSelected = true;
    Object.keys(fileInputs).forEach(view => {
      if (!fileInputs[view].files[0]) {
        allFilesSelected = false;
        uploadBoxes[view].style.borderColor = 'var(--danger)';
        uploadBoxes[view].style.backgroundColor = 'rgba(247, 37, 133, 0.05)';
      }
    });

    if (!allFilesSelected) {
      alert('Please upload all 4 views of the insulator');
      return;
    }

    const formData = new FormData();
    formData.append('front', fileInputs.front.files[0]);
    formData.append('back', fileInputs.back.files[0]);
    formData.append('left', fileInputs.left.files[0]);
    formData.append('right', fileInputs.right.files[0]);

    spinner.style.display = 'block';
    resultsSection.style.display = 'none';

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      spinner.style.display = 'none';
      displayResults(data);
    })
    .catch(err => {
      spinner.style.display = 'none';
      alert('Error during analysis. Please try again.');
      console.error(err);
    });
  });

  function displayResults(data) {
    const isDefective = data.overall_status === 'Defective';
    statusCard.className = isDefective ? 'status-card status-defective' : 'status-card status-normal';
    statusCard.querySelector('.status-icon i').className = isDefective ? 'fas fa-exclamation-circle' : 'fas fa-check-circle';
    statusCard.querySelector('.status-title').textContent = `Insulator Status: ${data.overall_status}`;
    statusCard.querySelector('.status-desc').textContent = isDefective ?
      'Defects detected in one or more views' : 'No defects detected in the analyzed views';

    resultsGrid.innerHTML = '';
    data.views.forEach(view => {
      const isViewDefective = view.prediction === 'Defective';
      const resultCard = document.createElement('div');
      resultCard.className = 'result-card';
      resultCard.innerHTML = `
        <img src="${view.visualization}" alt="${view.view} view analysis">
        <div class="result-info">
          <div class="result-title">
            <span class="result-view">${view.view} View</span>
            <span class="result-status ${isViewDefective ? 'status-defective-badge' : 'status-normal-badge'}">
              ${view.prediction}
            </span>
          </div>
          <p class="result-confidence">Confidence: ${view.confidence.toFixed(1)}%</p>
        </div>
      `;
      resultsGrid.appendChild(resultCard);
    });

    combinedVisualization.src = `${data.combined_visualization}?t=${new Date().getTime()}`;
    combinedVisualization.alt = 'Combined defect visualization';

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  newAnalysisBtn?.addEventListener('click', function () {
    uploadForm.reset();
    previewContainer.style.display = 'none';
    previewContainer.innerHTML = '';
    resultsSection.style.display = 'none';
    Object.keys(uploadBoxes).forEach(view => {
      uploadBoxes[view].style.borderColor = 'var(--light-gray)';
      uploadBoxes[view].style.backgroundColor = 'transparent';
    });
    uploadForm.scrollIntoView({ behavior: 'smooth' });
  });
});
