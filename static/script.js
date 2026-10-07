document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropZonePrompt = document.getElementById('dropZonePrompt');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const previewMeta = document.getElementById('previewMeta');
    const uploadForm = document.getElementById('uploadForm');
    const classifyBtn = document.getElementById('classifyBtn');
    const resetBtn = document.getElementById('resetBtn');

    // Result elements
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    
    const predClassName = document.getElementById('predClassName');
    const predSubtext = document.getElementById('predSubtext');
    const predConfidence = document.getElementById('predConfidence');
    const predIcon = document.getElementById('predIcon');
    const probBarsList = document.getElementById('probBarsList');

    const presetButtons = document.querySelectorAll('.preset-btn');

    let currentFile = null;

    const classConfig = {
        'Bus': { icon: 'fa-bus', color: 'var(--color-bus)', label: 'Public Transit Bus', desc: 'Heavy Commercial Transit' },
        'Car': { icon: 'fa-car-side', color: 'var(--color-car)', label: 'Passenger Car', desc: 'Light Private Transport' },
        'Motorcycle': { icon: 'fa-motorcycle', color: 'var(--color-motorcycle)', label: 'Motorcycle / Bike', desc: 'Two-Wheeled Motorized' },
        'Truck': { icon: 'fa-truck', color: 'var(--color-truck)', label: 'Cargo Truck', desc: 'Heavy Freight Transport' }
    };

    // Drag and Drop listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    resetBtn.addEventListener('click', () => {
        clearSelectedFile();
        showEmptyState();
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, JPEG, WEBP).');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZonePrompt.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            previewMeta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedFile() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZonePrompt.classList.remove('hidden');
        previewWrapper.classList.add('hidden');
        classifyBtn.disabled = true;
    }

    function showEmptyState() {
        emptyState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsContent.classList.add('hidden');
    }

    function showLoadingState() {
        emptyState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        classifyBtn.disabled = true;
    }

    function showResultsState() {
        emptyState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsContent.classList.remove('hidden');
        classifyBtn.disabled = false;
    }

    // Preset button handlers: fetch sample image or create canvas sample
    presetButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const cls = btn.getAttribute('data-class');
            // Try fetching from server static or sample image
            try {
                const response = await fetch(`/api/sample/${cls}`);
                if (response.ok) {
                    const blob = await response.blob();
                    const file = new File([blob], `sample_${cls.toLowerCase()}.jpg`, { type: 'image/jpeg' });
                    handleSelectedFile(file);
                    // auto trigger submit
                    setTimeout(() => uploadForm.dispatchEvent(new Event('submit')), 300);
                } else {
                    // Fallback to trigger file input browse
                    fileInput.click();
                }
            } catch (err) {
                fileInput.click();
            }
        });
    });

    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        showLoadingState();

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during prediction.');
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error('Prediction Error:', error);
            alert(`Classification failed: ${error.message}`);
            showEmptyState();
        }
    });

    function renderResults(data) {
        loadingState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const pred = data.prediction;
        const conf = data.confidence;
        const probs = data.probabilities;
        const cfg = classConfig[pred] || { icon: 'fa-car', color: 'var(--accent-blue)', label: pred, desc: 'Vehicle' };

        // Top prediction banner
        predClassName.textContent = pred;
        predSubtext.textContent = cfg.desc;
        predConfidence.textContent = `${conf}%`;
        predIcon.className = `fa-solid ${cfg.icon}`;

        // Render probability distribution bars
        probBarsList.innerHTML = '';
        const sortedClasses = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);

        sortedClasses.forEach(clsName => {
            const val = probs[clsName];
            const itemCfg = classConfig[clsName] || { icon: 'fa-car', color: '#94a3b8' };
            const isTop = (clsName === pred);

            const row = document.createElement('div');
            row.className = 'prob-row';

            row.innerHTML = `
                <div class="prob-meta">
                    <span class="prob-class-label" style="color: ${isTop ? '#ffffff' : 'var(--text-muted)'}; font-weight: ${isTop ? '700' : '500'}">
                        <i class="fa-solid ${itemCfg.icon}" style="color: ${itemCfg.color}"></i> ${clsName}
                        ${isTop ? '<span style="font-size: 0.68rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 1px 6px; border-radius: 4px; margin-left: 6px;">TOP MATCH</span>' : ''}
                    </span>
                    <span class="prob-pct-label" style="color: ${isTop ? itemCfg.color : 'var(--text-muted)'}; font-weight: ${isTop ? '700' : '500'}">
                        ${val}%
                    </span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: 0%; background: ${itemCfg.color}; color: ${itemCfg.color};"></div>
                </div>
            `;

            probBarsList.appendChild(row);

            // Animate bar width smoothly
            setTimeout(() => {
                const fill = row.querySelector('.prob-bar-fill');
                if (fill) fill.style.width = `${Math.max(val, 2)}%`;
            }, 60);
        });
    }
});
