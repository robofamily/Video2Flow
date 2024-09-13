import torch
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer

frames = iio.imread('./original.mp4', plugin="FFMPEG")  # plugin="pyav"
device = 'cuda'
grid_size = 100
video_chunk_max_len = 10
threshold_perc = 0.09
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
video_chunks = [video[:, ind:ind + video_chunk_max_len] for ind in range(0, video.shape[1], video_chunk_max_len)]

B = video.shape[0]
pred = {'tracks': [[] for i in range(B)], 'visibility': [[] for i in range(B)]}
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(device)
for video_chunk in video_chunks:
    cotracker(video_chunk=video_chunk, is_first_step=True, grid_size=grid_size)
    # Process the video
    for ind in range(0, video_chunk.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video_chunk[:, ind : ind + cotracker.step * 2]
        )  # B T N 2,  B T N
        B, T, N, _ = pred_tracks.shape
    diff_l2 = ((pred_tracks[:, 1:] - pred_tracks[:, :-1]) ** 2).sum(dim=(1, 3), keepdim=True) # B 1 N 1
    threshold = diff_l2.max(dim=2)[0] * threshold_perc # B 1 1
    for batch_idx in range(B):
        num_keep = (diff_l2[batch_idx] > threshold[batch_idx]).sum()
        print('num keep', num_keep.cpu().item())
        pred['tracks'][batch_idx].append(pred_tracks[batch_idx][diff_l2[batch_idx].repeat(T, 1, 2) > threshold[batch_idx]].view(T, num_keep, 2))
        pred['visibility'][batch_idx].append(pred_visibility[batch_idx][diff_l2[batch_idx, :, :, 0].repeat(T, 1) > threshold[batch_idx]].view(T, num_keep))

vis = Visualizer(
    save_dir="./saved_videos", 
    linewidth=1,
    mode='cool',
    tracks_leave_trace=-1,
)
track_videos = []
for chunk_id, video_chunk in enumerate(video_chunks):
    track_video = vis.visualize(
        video_chunk, 
        pred['tracks'][0][chunk_id].unsqueeze(0), 
        pred['visibility'][0][chunk_id].unsqueeze(0),
        save_video=False,
    )
    track_videos.append(track_video)
vis.save_video(
    torch.cat(track_videos, dim=1),
    filename='track',
    writer=None,
    step=0
)
