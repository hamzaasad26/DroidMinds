document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const previewContainer = document.querySelector('.preview-container');
    const previewImage = document.getElementById('preview-image');
    const removePreview = document.querySelector('.remove-preview');
    const browseBtn = document.querySelector('.browse-btn');
    const classifyBtn = document.querySelector('.classify-btn');
    const uploadContent = document.querySelector('.upload-content');
    const loadingSpinner = document.querySelector('.loading-spinner');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle file selection via button
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                displayPreview(file);
                classifyBtn.disabled = false;
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.hidden = false;
            uploadContent.hidden = true;
        }
        reader.readAsDataURL(file);
    }

    // Remove preview
    removePreview.addEventListener('click', function(e) {
        e.stopPropagation();
        clearPreview();
    });

    function clearPreview() {
        previewImage.src = '';
        previewContainer.hidden = true;
        uploadContent.hidden = false;
        fileInput.value = '';
        classifyBtn.disabled = true;
    }

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (fileInput.files.length > 0) {
            loadingSpinner.hidden = false;
            classifyBtn.disabled = true;
            
            // Simulate form submission (replace with actual form submission)
            setTimeout(() => {
                loadingSpinner.hidden = true;
                classifyBtn.disabled = false;
                // Add your actual form submission logic here
            }, 2000);
        }
    });

    // Click anywhere in drop zone to trigger file input
    dropZone.addEventListener('click', function(e) {
        if (e.target !== removePreview && !removePreview.contains(e.target)) {
            fileInput.click();
        }
    });
});