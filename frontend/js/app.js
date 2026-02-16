// EduShield - Frontend Application

const API_BASE_URL = 'https://edushield.onrender.com';

// Global state
let currentPage = 1;
let currentFilter = '';
let currentSearch = '';
let charts = {};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupNavigation();
    setupEventListeners();
    loadDashboard();
}

// Navigation
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const section = this.dataset.section;
            switchSection(section);
        });
    });
}

function switchSection(section) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.section === section) {
            btn.classList.add('active');
        }
    });
    
    // Update sections
    document.querySelectorAll('.section').forEach(sec => {
        sec.classList.remove('active');
    });
    document.getElementById(`${section}-section`).classList.add('active');
    
    // Load section data
    switch(section) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'students':
            loadStudents();
            break;
        case 'alerts':
            loadAlerts();
            break;
        case 'model':
            loadModelInfo();
            break;
    }
}

// Event Listeners
function setupEventListeners() {
    // Search
    document.getElementById('search-input').addEventListener('input', debounce(function(e) {
        currentSearch = e.target.value;
        currentPage = 1;
        loadStudents();
    }, 500));
    
    // Filter
    document.getElementById('risk-filter').addEventListener('change', function(e) {
        currentFilter = e.target.value;
        currentPage = 1;
        loadStudents();
    });
    
    // Intervention form
    document.getElementById('intervention-form').addEventListener('submit', handleInterventionSubmit);
}

// Dashboard
async function loadDashboard() {
    showSpinner();
    
    try {
        // Load stats
        const stats = await fetchAPI('/stats');
        updateStats(stats);
        
        // Load model info
        const modelInfo = await fetchAPI('/model-info');
        updateCharts(stats, modelInfo);
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showError('Failed to load dashboard data');
    } finally {
        hideSpinner();
    }
}

function updateStats(stats) {
    document.getElementById('total-students').textContent = stats.total_students;
    document.getElementById('high-risk-count').textContent = stats.high_risk_count;
    document.getElementById('medium-risk-count').textContent = stats.medium_risk_count;
    document.getElementById('low-risk-count').textContent = stats.low_risk_count;
}

function updateCharts(stats, modelInfo) {
    // Risk Distribution Chart
    const riskCtx = document.getElementById('risk-chart').getContext('2d');
    
    if (charts.riskChart) {
        charts.riskChart.destroy();
    }
    
    charts.riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [stats.low_risk_count, stats.medium_risk_count, stats.high_risk_count],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Metrics Chart
    const metricsCtx = document.getElementById('metrics-chart').getContext('2d');
    
    if (charts.metricsChart) {
        charts.metricsChart.destroy();
    }
    
    charts.metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            datasets: [{
                label: 'Score (%)',
                data: [
                    (modelInfo.accuracy * 100).toFixed(2),
                    (modelInfo.precision * 100).toFixed(2),
                    (modelInfo.recall * 100).toFixed(2),
                    (modelInfo.f1_score * 100).toFixed(2),
                    (modelInfo.roc_auc * 100).toFixed(2)
                ],
                backgroundColor: '#2563eb',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Students
async function loadStudents() {
    showSpinner();
    
    try {
        const skip = (currentPage - 1) * 50;
        const limit = 50;
        
        let url = `/students?skip=${skip}&limit=${limit}`;
        if (currentFilter) {
            url += `&risk_filter=${encodeURIComponent(currentFilter)}`;
        }
        
        const data = await fetchAPI(url);
        
        // Filter by search locally (for simplicity)
        let students = data.students;
        if (currentSearch) {
            const search = currentSearch.toLowerCase();
            students = students.filter(s => 
                s.student_id.toLowerCase().includes(search) ||
                s.student_name.toLowerCase().includes(search) ||
                (s.department && s.department.toLowerCase().includes(search))
            );
        }
        
        displayStudents(students);
        updatePagination();
        
    } catch (error) {
        console.error('Error loading students:', error);
        showError('Failed to load students');
    } finally {
        hideSpinner();
    }
}

function displayStudents(students) {
    const tbody = document.getElementById('students-tbody');
    
    if (students.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No students found</td></tr>';
        return;
    }
    
    tbody.innerHTML = students.map(student => `
        <tr>
            <td>${student.student_id}</td>
            <td>${student.student_name}</td>
            <td>${student.department || 'N/A'}</td>
            <td>${student.avg_score ? student.avg_score.toFixed(1) : '0.0'}</td>
            <td>${student.num_assessments || 0}</td>
            <td>${(student.current_risk_score * 100).toFixed(1)}%</td>
            <td>
                <span class="risk-badge ${getRiskClass(student.risk_category)}">
                    ${student.risk_emoji} ${student.risk_category}
                </span>
            </td>
            <td>
                <button class="btn btn-secondary" onclick="viewStudent('${student.student_id}')">View</button>
                <button class="btn btn-primary" onclick="scheduleIntervention('${student.student_id}', '${student.student_name}')">Intervene</button>
            </td>
        </tr>
    `).join('');
}

function getRiskClass(category) {
    if (category === 'High Risk') return 'high';
    if (category === 'Medium Risk') return 'medium';
    return 'low';
}

function updatePagination() {
    document.getElementById('page-info').textContent = `Page ${currentPage}`;
}

function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        loadStudents();
    }
}

function nextPage() {
    currentPage++;
    loadStudents();
}

// Student Details
async function viewStudent(studentId) {
    showSpinner();
    
    try {
        const data = await fetchAPI(`/students/${studentId}`);
        displayStudentDetails(data);
        openModal();
        
    } catch (error) {
        console.error('Error loading student details:', error);
        showError('Failed to load student details');
    } finally {
        hideSpinner();
    }
}

function displayStudentDetails(data) {
    const student = data.student;
    const predictions = data.prediction_history;
    const interventions = data.interventions;
    
    const content = document.getElementById('student-detail-content');
    
    content.innerHTML = `
        <div class="student-details">
            <div class="detail-section">
                <h3>📋 Basic Information</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <strong>Student ID:</strong> ${student.student_id}
                    </div>
                    <div class="detail-item">
                        <strong>Name:</strong> ${student.student_name}
                    </div>
                    <div class="detail-item">
                        <strong>Department:</strong> ${student.department || 'N/A'}
                    </div>
                    <div class="detail-item">
                        <strong>Gender:</strong> ${student.gender || 'N/A'}
                    </div>
                    <div class="detail-item">
                        <strong>Age:</strong> ${student.age_band || 'N/A'}
                    </div>
                    <div class="detail-item">
                        <strong>Education:</strong> ${student.education_level || 'N/A'}
                    </div>
                </div>
            </div>
            
            <div class="detail-section">
                <h3>📊 Academic Performance</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <strong>Average Score:</strong> ${student.avg_score ? student.avg_score.toFixed(1) : '0.0'}
                    </div>
                    <div class="detail-item">
                        <strong>Assessments:</strong> ${student.num_assessments || 0}
                    </div>
                    <div class="detail-item">
                        <strong>Current Risk:</strong> 
                        <span class="risk-badge ${getRiskClass(student.risk_category)}">
                            ${student.risk_emoji} ${(student.current_risk_score * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="detail-section">
                <h3>📈 Risk History</h3>
                ${predictions.length > 0 ? `
                    <canvas id="student-risk-chart"></canvas>
                ` : '<p class="text-muted">No prediction history available</p>'}
            </div>
            
            <div class="detail-section">
                <h3>🎯 Interventions (${interventions.length})</h3>
                ${interventions.length > 0 ? `
                    <div class="interventions-list">
                        ${interventions.map(i => `
                            <div class="intervention-item">
                                <strong>${i.intervention_type}</strong>
                                <p>${i.description}</p>
                                <small>Scheduled: ${new Date(i.scheduled_date).toLocaleDateString()}</small>
                                <span class="status-badge">${i.status}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : '<p class="text-muted">No interventions scheduled</p>'}
            </div>
        </div>
    `;
    
    // Draw risk history chart if data available
    if (predictions.length > 0) {
        setTimeout(() => {
            const ctx = document.getElementById('student-risk-chart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: predictions.map((_, i) => `Prediction ${i + 1}`).reverse(),
                    datasets: [{
                        label: 'Risk Percentage',
                        data: predictions.map(p => p.risk_percentage).reverse(),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }, 100);
    }
}

function openModal() {
    document.getElementById('student-modal').classList.add('active');
}

function closeModal() {
    document.getElementById('student-modal').classList.remove('active');
}

// Alerts
async function loadAlerts() {
    showSpinner();
    
    try {
        const data = await fetchAPI('/alerts?threshold=0.7');
        displayAlerts(data.alerts);
        
    } catch (error) {
        console.error('Error loading alerts:', error);
        showError('Failed to load alerts');
    } finally {
        hideSpinner();
    }
}

function displayAlerts(alerts) {
    const container = document.getElementById('alerts-container');
    
    if (alerts.length === 0) {
        container.innerHTML = '<p class="text-center text-muted">No high-risk alerts at this time</p>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => `
        <div class="alert-card">
            <div class="alert-header">
                <div>
                    <div class="alert-student">${alert.student_name}</div>
                    <div class="alert-details">ID: ${alert.student_id}</div>
                </div>
                <div class="alert-risk">${(alert.current_risk_score * 100).toFixed(0)}%</div>
            </div>
            
            <div class="alert-details">
                <strong>Department:</strong> ${alert.department || 'N/A'}<br>
                <strong>Avg Score:</strong> ${alert.avg_score ? alert.avg_score.toFixed(1) : 'N/A'}<br>
                <strong>Assessments:</strong> ${alert.num_assessments || 0}
            </div>
            
            <div class="alert-factors">
                <h4>Recent Risk Trend:</h4>
                ${alert.recent_predictions && alert.recent_predictions.length > 0 ? 
                    alert.recent_predictions.map(r => `<span class="factor-tag">${r.toFixed(1)}%</span>`).join('') 
                    : '<span class="factor-tag">New student</span>'}
            </div>
            
            <div class="mt-20">
                <button class="btn btn-primary" style="width: 100%;" 
                        onclick="scheduleIntervention('${alert.student_id}', '${alert.student_name}')">
                    📅 Schedule Intervention
                </button>
            </div>
        </div>
    `).join('');
}

// Interventions
function scheduleIntervention(studentId, studentName) {
    document.getElementById('intervention-student-id').value = studentId;
    document.getElementById('intervention-student-name').value = studentName;
    document.getElementById('intervention-modal').classList.add('active');
}

function closeInterventionModal() {
    document.getElementById('intervention-modal').classList.remove('active');
    document.getElementById('intervention-form').reset();
}

async function handleInterventionSubmit(e) {
    e.preventDefault();
    showSpinner();
    
    try {
        const formData = {
            student_id: document.getElementById('intervention-student-id').value,
            intervention_type: document.getElementById('intervention-type').value,
            description: document.getElementById('intervention-description').value,
            scheduled_date: document.getElementById('intervention-date').value,
            assigned_to: document.getElementById('intervention-assigned').value || null
        };
        
        await fetchAPI('/intervention', {
            method: 'POST',
            body: JSON.stringify(formData)
        });
        
        showSuccess('Intervention scheduled successfully!');
        closeInterventionModal();
        loadAlerts(); // Refresh alerts
        
    } catch (error) {
        console.error('Error scheduling intervention:', error);
        showError('Failed to schedule intervention');
    } finally {
        hideSpinner();
    }
}

// Model Info
async function loadModelInfo() {
    showSpinner();
    
    try {
        const modelInfo = await fetchAPI('/model-info');
        displayModelInfo(modelInfo);
        
    } catch (error) {
        console.error('Error loading model info:', error);
        showError('Failed to load model information');
    } finally {
        hideSpinner();
    }
}

function displayModelInfo(modelInfo) {
    // Display metrics
    const metricsHtml = `
        <div class="metric-item">
            <span class="metric-label">Accuracy</span>
            <span class="metric-value">${(modelInfo.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Precision</span>
            <span class="metric-value">${(modelInfo.precision * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Recall</span>
            <span class="metric-value">${(modelInfo.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">F1-Score</span>
            <span class="metric-value">${(modelInfo.f1_score * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">ROC-AUC</span>
            <span class="metric-value">${(modelInfo.roc_auc * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Students Trained</span>
            <span class="metric-value">${modelInfo.total_students_trained.toLocaleString()}</span>
        </div>
    `;
    
    document.getElementById('model-metrics').innerHTML = metricsHtml;
    
    // Feature importance chart - destroy and recreate safely
    try {
        if (charts.featureChart) {
            charts.featureChart.destroy();
            charts.featureChart = null;
        }
        
        const featureCtx = document.getElementById('feature-chart').getContext('2d');
        const topFeatures = modelInfo.top_features.slice(0, 10);
        
        charts.featureChart = new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: topFeatures.map(f => f.feature),
                datasets: [{
                    label: 'Importance',
                    data: topFeatures.map(f => (f.importance * 100).toFixed(2)),
                    backgroundColor: '#2563eb'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } catch (err) {
        console.error('Error creating feature chart:', err);
    }
}

async function trainModel() {
    if (!confirm('Training the model will take several minutes. Continue?')) {
        return;
    }
    
    showSpinner();
    const logsContainer = document.getElementById('training-logs');
    logsContainer.innerHTML = 'Training started...\n';
    
    try {
        const result = await fetchAPI('/train-model', { method: 'POST' });
        
        logsContainer.innerHTML += '\n✅ Training completed successfully!\n';
        logsContainer.innerHTML += `\nAccuracy: ${(result.metrics.accuracy * 100).toFixed(2)}%\n`;
        logsContainer.innerHTML += `Precision: ${(result.metrics.precision * 100).toFixed(2)}%\n`;
        logsContainer.innerHTML += `Recall: ${(result.metrics.recall * 100).toFixed(2)}%\n`;
        
        showSuccess('Model trained successfully!');
        loadModelInfo();
        
    } catch (error) {
        console.error('Error training model:', error);
        logsContainer.innerHTML += '\n❌ Training failed!\n';
        showError('Failed to train model');
    } finally {
        hideSpinner();
    }
}

// Utility Functions
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    const response = await fetch(url, { ...defaultOptions, ...options });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

function showSpinner() {
    document.getElementById('loading-spinner').classList.remove('hidden');
}

function hideSpinner() {
    document.getElementById('loading-spinner').classList.add('hidden');
}

function showError(message) {
    alert('❌ ' + message);
}

function showSuccess(message) {
    alert('✅ ' + message);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}