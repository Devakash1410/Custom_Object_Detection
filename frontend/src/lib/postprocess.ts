/**
 * Non-Maximum Suppression (NMS) — JavaScript implementation.
 * Filters overlapping detections per class.
 */

import type { Detection } from "@/types/detection";

/**
 * Compute Intersection-over-Union between two bounding boxes.
 */
function computeIoU(a: Detection, b: Detection): number {
  const x1 = Math.max(a.bbox.x1, b.bbox.x1);
  const y1 = Math.max(a.bbox.y1, b.bbox.y1);
  const x2 = Math.min(a.bbox.x2, b.bbox.x2);
  const y2 = Math.min(a.bbox.y2, b.bbox.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (intersection === 0) return 0;

  const areaA = a.bbox.width * a.bbox.height;
  const areaB = b.bbox.width * b.bbox.height;
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

/**
 * Apply per-class NMS to filter overlapping detections.
 */
export function applyNMS(
  detections: Detection[],
  iouThreshold: number = 0.45
): Detection[] {
  if (detections.length === 0) return [];

  // Group by class
  const byClass = new Map<number, Detection[]>();
  for (const det of detections) {
    const list = byClass.get(det.classId) || [];
    list.push(det);
    byClass.set(det.classId, list);
  }

  const result: Detection[] = [];

  for (const [, classDets] of byClass) {
    // Sort by confidence descending
    classDets.sort((a, b) => b.confidence - a.confidence);

    const kept: boolean[] = new Array(classDets.length).fill(true);

    for (let i = 0; i < classDets.length; i++) {
      if (!kept[i]) continue;

      result.push(classDets[i]);

      for (let j = i + 1; j < classDets.length; j++) {
        if (!kept[j]) continue;
        if (computeIoU(classDets[i], classDets[j]) > iouThreshold) {
          kept[j] = false;
        }
      }
    }
  }

  // Sort final result by confidence
  result.sort((a, b) => b.confidence - a.confidence);
  return result;
}
