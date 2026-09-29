import { describe, expect, it } from 'vitest';
import { computeRiskScore, DEFAULT_ARBITRATION_CONFIG, routeEvent } from './math';

const AGREE = 'multi_sensor_cross_confirmed' as const;
const DISAGREE = 'disagreement' as const;

describe('routeEvent', () => {
  it('matches the backend class1_threshold (0.60), not the old 0.45', () => {
    expect(DEFAULT_ARBITRATION_CONFIG.class1_threshold).toBe(0.6);
    // Below 0.60: not a strong-model escalation on its own.
    expect(routeEvent({ 1: 0.5, 5: 0.5 }, 0.1, false, 0, AGREE, 0.9)).not.toBe('CRITICAL');
    // At/above 0.60: escalates.
    expect(routeEvent({ 1: 0.6, 5: 0.4 }, 0.1, false, 0, AGREE, 0.9)).toBe('CRITICAL');
  });

  it('routes sensor disagreement to UNCERTAIN regardless of model confidence', () => {
    expect(routeEvent({ 1: 0.99 }, 0.1, false, 0, DISAGREE, 0.9)).toBe('UNCERTAIN');
  });

  it('never auto-escalates an unusual-industrial-fire call on water land cover', () => {
    const onWater = routeEvent({ 1: 0.95 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, true, true, true);
    expect(onWater).toBe('UNCERTAIN');
    const onLand = routeEvent({ 1: 0.95 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, true, true, false);
    expect(onLand).toBe('CRITICAL');
  });

  it('routes incomplete 90-day history to UNCERTAIN even with a confident call', () => {
    const incomplete = routeEvent(
      { 1: 0.95 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, true, false, false,
    );
    expect(incomplete).toBe('UNCERTAIN');
  });

  it('routes out-of-vocabulary land cover to UNCERTAIN when the call is not already critical on its own', () => {
    // p1 = 0.58 clears min_model_confidence (0.55) but not class1_threshold (0.60), so a
    // confident-on-its-own escalation never enters the picture and the vocabulary gate decides.
    const outOfVocab = routeEvent(
      { 1: 0.58, 5: 0.42 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, false, true, false,
    );
    expect(outOfVocab).toBe('UNCERTAIN');
    const inVocab = routeEvent(
      { 1: 0.58, 5: 0.42 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, true, true, false,
    );
    expect(inVocab).toBe('NORMAL');
  });

  it('lets a confident model call escalate even when land cover is out of vocabulary', () => {
    // Matches backend/pipeline/arbitration.py: `critical` is checked before `uncertain`.
    const state = routeEvent(
      { 1: 0.95 }, 0.1, false, 0, AGREE, 0.9, DEFAULT_ARBITRATION_CONFIG, false, true, false,
    );
    expect(state).toBe('CRITICAL');
  });

  it('escalates an industrial-anomaly call on adequate-quality data', () => {
    const state = routeEvent({ 5: 0.9 }, 0.1, true, 5.0, AGREE, 0.9);
    expect(state).toBe('CRITICAL');
  });

  it('falls back to NORMAL when nothing crosses a threshold', () => {
    const state = routeEvent({ 5: 0.9 }, 0.1, false, 0, AGREE, 0.9);
    expect(state).toBe('NORMAL');
  });
});

describe('computeRiskScore', () => {
  it('produces a total in [0, 100] and clamps exposure', () => {
    const risk = computeRiskScore({ 1: 1.0 }, 5.0, 8, 20, 1.5);
    expect(risk.total).toBeGreaterThanOrEqual(0);
    expect(risk.total).toBeLessThanOrEqual(100);
    expect(risk.exposure).toBe(1);
  });
});
