import numpy as np

CLASSLIST = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
             'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
             'Shoplifting', 'Stealing', 'Vandalism']


def str2ind(categoryname, classlist):
    for i in range(len(classlist)):
        if categoryname == classlist[i]:
            return i


def nms(dets, thresh=0.6, top_k=-1):
    if len(dets) == 0:
        return [], []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep], keep


def getLocMAP(predictions, th, gtsegments, gtlabels, excludeNormal):
    if excludeNormal:
        classes_num = 13
        videos_num = 140
        predictions = predictions[:videos_num]
    else:
        classes_num = 14
        videos_num = 290

    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = -p
        [pp[:, i].sort() for i in range(np.shape(pp)[1])]
        pp = -pp
        c_s = np.mean(pp[:int(np.shape(pp)[0] / 16), :], axis=0)
        ind = c_s > 0.0
        c_score.append(c_s)
        predictions_mod.append(p * ind)
    predictions = predictions_mod

    ap = []
    for c in range(0, 14):
        segment_predict = []
        for i in range(len(predictions)):
            tmp = predictions[i][:, c]
            segment_predict_multithr = []
            thr_set = np.arange(0.6, 0.7, 0.1)
            for thr in thr_set:
                threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thr
                vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
                vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
                for j in range(len(s)):
                    if e[j] - s[j] >= 2:
                        segment_scores = np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]
                        segment_predict_multithr.append([i, s[j], e[j], segment_scores])
            if len(segment_predict_multithr) != 0:
                segment_predict_multithr = np.array(segment_predict_multithr)
                segment_predict_multithr = segment_predict_multithr[np.argsort(-segment_predict_multithr[:, -1])]
                _, keep = nms(segment_predict_multithr[:, 1:-1], 0.6)
                segment_predict.extend(list(segment_predict_multithr[keep]))
        segment_predict = np.array(segment_predict)

        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments))
                      for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j], CLASSLIST) == c]
        gtpos = len(segment_gt)

        tp, fp = [], []
        for i in range(len(segment_predict)):
            flag = 0.
            best_iou = 0.0
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(len(set(gt).union(set(p))))
                    if IoU >= th:
                        flag = 1.
                        if IoU > best_iou:
                            best_iou = IoU
                            best_j = j
            if flag > 0:
                del segment_gt[best_j]
            tp.append(flag)
            fp.append(1. - flag)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp) == 0:
            prc = 0.
        else:
            prc = np.sum((tp_c / (fp_c + tp_c)) * tp) / gtpos
        ap.append(prc)
    return 100 * np.mean(ap)


def getDetectionMAP(predictions, segments, labels, excludeNormal=False):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = []
    for iou in iou_list:
        dmap_list.append(getLocMAP(predictions, iou, segments, labels, excludeNormal))
    return dmap_list, iou_list


def _loc_map_agnostic(predictions, th, gtsegments, gtlabels):
    """
    Class-agnostic temporal localization mAP at a single IoU threshold.
    predictions: list of 1-D np.ndarray (per-frame score)
    gtsegments: list of list of [start_frame, end_frame]
    gtlabels: list of list of class names (ignored; kept for interface parity)
    Returns: float AP % (0..100)
    """
    videos_num = len(predictions)

    predictions_mod = []
    c_score = []
    for p in predictions:
        # p: 1-D [n_frames]; keep top-k% frame scores as video-level "actionness"
        pp = np.sort(p)[::-1]  # descending
        c_s = np.mean(pp[:max(1, int(len(pp) / 16))])
        c_score.append(c_s)
        predictions_mod.append(p)
    predictions = predictions_mod

    segment_predict = []
    for i in range(videos_num):
        tmp = predictions[i]
        segment_predict_multithr = []
        thr_set = np.arange(0.6, 0.7, 0.1)
        for thr in thr_set:
            if tmp.max() == tmp.min():
                continue
            threshold = tmp.max() - (tmp.max() - tmp.min()) * thr
            vid_pred = np.concatenate([np.zeros(1),
                                       (tmp > threshold).astype('float32'),
                                       np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1]
                             for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                if e[j] - s[j] >= 2:
                    segment_scores = float(np.max(tmp[s[j]:e[j]])) + 0.7 * c_score[i]
                    segment_predict_multithr.append([i, s[j], e[j], segment_scores])
        if len(segment_predict_multithr) != 0:
            arr = np.array(segment_predict_multithr)
            arr = arr[np.argsort(-arr[:, -1])]
            _, keep = nms(arr[:, 1:-1], 0.6)
            segment_predict.extend(list(arr[keep]))

    segment_predict = np.array(segment_predict) if len(segment_predict) else np.zeros((0, 4))
    if len(segment_predict) == 0:
        return 0.0
    segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

    # Collapse ALL GT segments (any class) into a single "anomaly" class
    segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]]
                  for i in range(len(gtsegments))
                  for j in range(len(gtsegments[i]))]
    gtpos = len(segment_gt)
    if gtpos == 0:
        return 0.0

    tp, fp = [], []
    for i in range(len(segment_predict)):
        flag = 0.0
        best_iou = 0.0
        best_j = -1
        for j in range(len(segment_gt)):
            if segment_predict[i][0] == segment_gt[j][0]:
                gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
                p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                inter = len(set(gt).intersection(set(p)))
                union = len(set(gt).union(set(p)))
                if union == 0:
                    continue
                IoU = float(inter) / float(union)
                if IoU >= th and IoU > best_iou:
                    flag = 1.0
                    best_iou = IoU
                    best_j = j
        if flag > 0 and best_j >= 0:
            del segment_gt[best_j]
        tp.append(flag)
        fp.append(1.0 - flag)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    if sum(tp) == 0:
        prc = 0.0
    else:
        prc = np.sum((tp_c / (fp_c + tp_c)) * np.array(tp)) / gtpos
    return 100.0 * prc


def getDetectionMAP_agnostic(predictions, gtsegments, gtlabels):
    """Class-agnostic version of getDetectionMAP.
    predictions: list of 1-D np.ndarray (frame-level scores).
    gtsegments, gtlabels: same structure as the class-aware variant.
    """
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = [_loc_map_agnostic(predictions, iou, gtsegments, gtlabels)
                 for iou in iou_list]
    return dmap_list, iou_list
