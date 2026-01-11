# SepsisPulse - Product Requirements Document

## Executive Summary

SepsisPulse is a clinical decision support tool that audits and compares three mathematical approaches to early sepsis prediction. The system evaluates models using the Clinical Utility Score from the PhysioNet Challenge 2019, which rewards early detection while penalizing false alarms and missed cases.

---

## Problem Statement

### The Clinical Challenge

**Sepsis is a medical emergency.** It is a life-threatening condition caused by the body's dysregulated response to infection, leading to organ dysfunction. Sepsis affects approximately 1.7 million adults in the United States annually and contributes to nearly 270,000 deaths per year.

**Time is critical.** Every hour of delayed sepsis treatment increases mortality by approximately 8%. Early recognition and intervention are essential for patient survival.

### The Alert Fatigue Crisis

Current hospital monitoring systems suffer from a phenomenon known as **Alarm Fatigue**:

- ICU nurses experience 150-400 alarms per patient per day
- 72-99% of clinical alarms are false positives
- Clinicians become desensitized and may ignore or delay response to critical alerts
- This leads to preventable adverse events and deaths

### The Prediction Timing Dilemma

Existing sepsis prediction systems face a fundamental trade-off:

| Prediction Type | Consequence |
|-----------------|-------------|
| Too Early | High false positive rate, contributes to alarm fatigue |
| Too Late | Patient deterioration, reduced treatment efficacy, increased mortality |
| Optimal Window | 6-12 hours before onset, allows intervention while maximizing accuracy |

---

## Product Objectives

### Primary Objective

Develop a clinical utility auditor that determines the **Optimal Alert Window** for sepsis prediction - identifying how early sepsis can be reliably detected without creating excessive false alarms.

### Secondary Objectives

1. **Compare Prediction Approaches**: Evaluate rule-based (qSOFA), traditional ML (XGBoost-TS), and deep learning (TFT-Lite) methods
2. **Clinical Utility Focus**: Prioritize the PhysioNet Clinical Utility Score over traditional ML metrics
3. **Lead Time Analysis**: Measure and visualize how early each model detects sepsis
4. **Accessibility**: Deploy as a free, resource-efficient web application

---

## Success Metrics

### Primary Metric: Clinical Utility Score

The Clinical Utility Score (from PhysioNet Challenge 2019) is the primary success metric:

| Prediction Timing | Utility Value |
|-------------------|---------------|
| Optimal Early (6-12h before onset) | +1.0 |
| Acceptable Late (0-6h before onset) | +0.5 |
| Too Early (>12h before onset) | -0.05 |
| Missed/After Onset | -2.0 |
| False Positive (non-sepsis patient) | -0.05 |

**Target**: Achieve normalized utility score > 0.40 with TFT-Lite model.

### Secondary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Mean Lead Time | > 6 hours | Sufficient time for clinical intervention |
| AUC-ROC | > 0.85 | Strong discriminative ability |
| Sensitivity | > 0.80 | Catch most sepsis cases |
| Specificity | > 0.85 | Minimize false alarms |

### Operational Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Memory Usage | < 2 GB | HuggingFace free tier compatibility |
| Inference Time | < 100ms per patient | Real-time clinical use |
| Page Load Time | < 3 seconds | Usable dashboard experience |

---

## User Stories

### Clinical Users

**As an ICU physician**, I want to see which sepsis prediction model provides the best balance of early detection and false alarm reduction, so that I can make informed decisions about implementing alerts in my unit.

**As a clinical informaticist**, I want to compare the Clinical Utility Score across different prediction approaches, so that I can recommend the most effective model for our hospital's EHR integration.

**As a quality improvement officer**, I want to analyze lead time distributions for sepsis predictions, so that I can quantify the potential impact on patient outcomes.

### Technical Users

**As an ML researcher**, I want to evaluate my custom sepsis prediction model against established baselines (qSOFA, XGBoost), so that I can benchmark my approach.

**As a data scientist**, I want to upload patient data in PhysioNet format and generate predictions from multiple models, so that I can analyze model behavior on my dataset.

**As a developer**, I want to understand the model architectures and their trade-offs, so that I can extend or adapt the system for other clinical prediction tasks.

---

## Non-Functional Requirements

### Resource Constraints

| Constraint | Specification | Rationale |
|------------|---------------|-----------|
| Maximum RAM | 2 GB | HuggingFace Spaces free tier limit |
| Disk Space | < 500 MB | Efficient deployment |
| GPU | Not Required | CPU-only inference |
| Internet | Required | Streamlit web application |

### Performance Requirements

| Requirement | Specification |
|-------------|---------------|
| Concurrent Users | 5+ simultaneous |
| Response Time | < 2 seconds for predictions |
| Data Processing | 1000 patients in < 30 seconds |
| Model Loading | < 5 seconds startup |

### Compatibility

| Platform | Support Level |
|----------|---------------|
| Python | 3.9+ |
| Browsers | Chrome, Firefox, Safari, Edge (latest) |
| Deployment | Streamlit Cloud, HuggingFace Spaces, Local |

### Data Privacy

- No patient data is stored server-side
- All processing occurs in-session
- Sample data is synthetic (not real patient data)
- PhysioNet data remains on user's system

---

## Scope

### In Scope

- Clinical utility score calculation and comparison
- Lead time analysis and visualization
- Three model implementations (qSOFA, XGBoost-TS, TFT-Lite)
- Pre-trained weights for XGBoost-TS and TFT-Lite
- Interactive Streamlit dashboard
- Sample synthetic patient data
- PhysioNet 2019 data format support

### Out of Scope

- Real-time EHR integration
- Model training pipeline (inference only)
- Clinical deployment certification (FDA/CE)
- Multi-language support
- Mobile-specific UI
- Patient-identifiable data storage

---

## Dependencies

### External Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| PhysioNet 2019 Data | Full dataset | Requires registration |
| Streamlit | Web framework | 1.28+ |
| PyTorch | Deep learning | 2.0+ |
| XGBoost | Gradient boosting | 2.0+ |

### Internal Dependencies

| Component | Depends On |
|-----------|------------|
| Dashboard | All models, evaluation modules |
| TFT-Lite | PyTorch, preprocessor |
| XGBoost-TS | XGBoost, feature engineering |
| Evaluation | Predictions from all models |

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Core | 2 weeks | Data loading, preprocessing, qSOFA baseline |
| Phase 2: ML Models | 2 weeks | XGBoost-TS, TFT-Lite implementation |
| Phase 3: Evaluation | 1 week | Clinical utility, lead time metrics |
| Phase 4: Dashboard | 1 week | Streamlit UI, visualizations |
| Phase 5: Documentation | 1 week | User guide, API docs, deployment |

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Memory exceeds 2GB | Cannot deploy to free tier | Medium | TFT-Lite architecture, quantization |
| PhysioNet access delays | Users cannot get full data | High | Pre-bundled sample subset |
| Model accuracy insufficient | Low clinical utility | Low | Multiple model comparison |
| Streamlit performance | Poor user experience | Low | Caching, optimized data loading |

---

## Appendix: Clinical Background

### Sepsis Definition (Sepsis-3)

Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection. Organ dysfunction is identified as an acute change in total SOFA score >= 2 points consequent to the infection.

### qSOFA Criteria

Quick SOFA (qSOFA) provides a bedside assessment for patients with suspected infection:

| Criterion | Threshold |
|-----------|-----------|
| Respiratory Rate | >= 22 breaths/min |
| Systolic Blood Pressure | <= 100 mmHg |
| Altered Mentation | GCS < 15 |

A qSOFA score >= 2 indicates high risk and prompts further assessment.

### References

1. Singer M, et al. "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." JAMA. 2016;315(8):801-810.
2. Reyna MA, et al. "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine. 2020;48(2):210-217.
3. Kumar A, et al. "Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock." Critical Care Medicine. 2006;34(6):1589-1596.
