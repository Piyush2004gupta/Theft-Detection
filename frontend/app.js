// ---------------------------------------------------------------------------
// API base URL – change this if your backend runs on a different host/port
// ---------------------------------------------------------------------------
const API_BASE = window.THEFT_API_BASE || `http://${window.location.hostname}:8000`;

const form = document.getElementById('analyze-form');
const videoInput = document.getElementById('video-input');
const saveVideoInput = document.getElementById('save-video');
const submitBtn = document.getElementById('submit-btn');
const statusBox = document.getElementById('status');

const overallStatus = document.getElementById('overall-status');
const totalPeople = document.getElementById('total-people');

const suspiciousIds = document.getElementById('suspicious-ids');
const rawJson = document.getElementById('raw-json');

const peopleTableBody = document.querySelector('#people-table tbody');
const resultVideo = document.getElementById('result-video');
const downloadLink = document.getElementById('download-link');

function setStatus(message, type) {
  statusBox.textContent = message;
  statusBox.className = `status ${type}`;
}

function setOverallBadge(status) {
  overallStatus.textContent = status;
  overallStatus.className = 'badge neutral';
  if (status === 'Normal') overallStatus.className = 'badge normal';
  if (status === 'Theft Detected') overallStatus.className = 'badge theft';
}

function renderPeople(people) {
  if (!Array.isArray(people) || people.length === 0) {
    peopleTableBody.innerHTML = '<tr><td colspan="5" class="empty">No people detected.</td></tr>';
    return;
  }
  peopleTableBody.innerHTML = people
    .map((person) => {
      const activityClass = person.activity === 'Theft' ? 'theft' : '';
      return `
        <tr>
          <td>${person.id}</td>
          <td>${person.in_time}</td>
          <td>${person.out_time}</td>
          <td>${person.time_spent_seconds}</td>
          <td class="${activityClass}">${person.activity}</td>
        </tr>
      `;
    })
    .join('');
}

function renderVideo(filename) {
  if (!filename) {
    resultVideo.removeAttribute('src');
    resultVideo.load();
    downloadLink.hidden = true;
    return;
  }
  const url = `${API_BASE}/processed/${encodeURIComponent(filename)}?t=${Date.now()}`;
  resultVideo.src = url;
  resultVideo.load();
  downloadLink.href = url;
  downloadLink.hidden = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = videoInput.files?.[0];
  if (!file) {
    setStatus('Please choose a video first.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  const saveVideo = saveVideoInput.checked;

  submitBtn.disabled = true;
  setStatus(`Analyzing "${file.name}"… this can take some time.`, 'loading');

  try {
    const url = `${API_BASE}/analyze?save_video=${saveVideo}`;
    console.log(`Sending request to: ${url}`);
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      const detail = data?.detail || 'Analysis failed.';
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }

    totalPeople.textContent = data.total_people ?? 0;
    suspiciousIds.textContent = JSON.stringify(data.suspicious_ids ?? []);
    setOverallBadge(data.overall_status ?? 'N/A');

    renderPeople(data.people || []);
    if (data.processed_video_path) {
      renderVideo(data.processed_video_path);
      setStatus('Analysis completed successfully.', 'success');
    } else {
      renderVideo(null);
      setStatus('Analysis complete. No video output available.', 'warning');
    }
    rawJson.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    console.error('Fetch error details:', error);
    setStatus(`Error: ${error.message}. Check browser console for details.`, 'error');
  } finally {
    submitBtn.disabled = false;
  }
});
