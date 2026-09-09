# Evaluation scenes by scene class

The evaluation split is **210 scenes / 1,097 episodes**, and every scene carries exactly one of the five
scene classes. This file lists which. It is generated from the published episode file by
`tools/make_scene_table.py`, so it cannot drift away from what a run is scored against.

Scene ids are the official ids of their source dataset. The directory layout each one is expected in is in the
Data section of the [README](README.md).

| Scene class | Scenes | Episodes | habitat_gs | hm3d | interiorgs | mp3d |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apartment | 120 | 239 | 1 | 39 | 68 | 12 |
| House | 61 | 216 | 0 | 31 | 2 | 28 |
| Commercial | 10 | 195 | 1 | 2 | 2 | 5 |
| Institution | 11 | 219 | 0 | 1 | 3 | 7 |
| Outdoor | 8 | 228 | 8 | 0 | 0 | 0 |
| **Total** | **210** | **1,097** | **10** | **73** | **75** | **52** |

## Apartment

single-storey flats: no interior staircase between levels. 120 scenes, 239 episodes.

**habitat_gs** (1)

<pre>scene63</pre>

**hm3d** (39)

<pre>00016-qk9eeNeR4vw  00019-AfKhsVmG8L4  00054-6BReaxZUoMg  00055-HxmXPBbFCkH  00078-nJTPfwbAj4S  00084-CtZLhCbWFm7  00104-KJxdMPgweZG  00108-oStKKWkQ1id  00185-NjyeoK5BLx3  00197-x4LVLSsYWcV  00253-HjxjHvpdeoM  00262-1xGrZPxG1Hz  00268-mHJxL9jnCox  00285-QKGMrurUVbk  00297-jGdNyKqGZJw  00310-WnvnMQh4eEa  00323-yHLr6bvWsVm  00332-cWfRoQnzNiM  00338-rK4jPRTUw15  00351-QxfX5te1gFu  00391-3UDjdrwcqMb  00404-QN2dRqwd84J  00426-X9fRPGxw1jS  00448-tAQTHnJ7n72  00480-RrfVebebfWf  00499-q6tn1ZjSsG4  00527-HPrcqBkKzuy  00591-JptJPosx1Z6  00593-m17UDpW3tHm  00594-1sPp3Wz8TCB  00637-iNpfPhK1sRz  00661-SSwbmq72C21  00708-eUJx9a4u63E  00733-GtM3JtRvvvR  00735-ypcVfePF8TG  00748-8QtyGUUtacf  00757-LVgQNuK8vtv  00795-awcRF7AZnJu  00797-99ML7CGPqsQ</pre>

**interiorgs** (68)

<pre>interior_0270_840784  interior_0342_840398  interior_0344_840395  interior_0403_840133  interior_0460_840546  interior_0464_840768  interior_0466_840779  interior_0470_840871  interior_0525_840837  interior_0527_840841  interior_0531_840848  interior_0533_840855  interior_0588_841009  interior_0591_841148  interior_0594_841158  interior_0601_841162  interior_0603_841172  interior_0658_841473  interior_0660_841475  interior_0662_841479  interior_0666_841484  interior_0667_841485  interior_0668_841486  interior_0717_841554  interior_0718_841558  interior_0720_841560  interior_0723_841566  interior_0724_841569  interior_0726_841575  interior_0727_841576  interior_0728_841577  interior_0781_840320  interior_0783_840266  interior_0784_840914  interior_0785_841229  interior_0786_841232  interior_0787_841244  interior_0788_841245  interior_0795_840845  interior_0796_840896  interior_0847_841768  interior_0849_841770  interior_0850_841771  interior_0851_841772  interior_0852_841773  interior_0854_841775  interior_0855_841776  interior_0856_841777  interior_0857_841779  interior_0859_841781  interior_0860_841782  interior_0861_841783  interior_0863_841785  interior_0913_841624  interior_0915_841635  interior_0917_840283  interior_0922_841863  interior_0923_841869  interior_0924_841874  interior_0928_841910  interior_0980_841921  interior_0981_841924  interior_0984_841961  interior_0986_841963  interior_0992_841977  interior_0994_841985  interior_0995_841988  interior_0996_841990</pre>

**mp3d** (12)

<pre>2t7WUuJeko7  5LpN3gDmAk7  7y3sRwLe3Va  GdvgFV5R1Z5  JF19kD82Mey  JeFG25nYj2p  RPmz2sHmrrY  UwV83HsGsw3  WYY7iVyf5p8  jh4fc5c5qoQ  kEZ7cmS4wCh  zsNo4HB9uLZ</pre>


## House

multi-storey or detached homes; an interior staircase is the deciding cue, with a yard or porch as support. Attics and basements count here too. 61 scenes, 216 episodes.

**hm3d** (31)

<pre>00001-UVdNNRcVyV1  00013-sfbj7jspYWj  00018-as8Y8AYx6yW  00037-oKFJo8jpzRW  00083-16tymPtM7uS  00105-xWvSkKiWQpC  00123-C3ifY177Ldq  00131-bZsfeA9uRk7  00153-28FFMGySc6D  00182-qWP3MMQM3eJ  00184-nzuiinFMXvf  00241-h6nwVLpAKQz  00270-bDTsgcSK5Qr  00287-SBHLgvFTVMZ  00298-by8SK9u18S8  00306-Y4L8fjz2yH7  00316-LqsTKpxKVP2  00364-tEafuWwhhwr  00367-RHdkyzXFp1k  00379-58fkJMgLopt  00414-77mMEyxhs44  00505-ZwnLFNzxASM  00512-WZDzPCybQvS  00565-xc2kFoo9nbw  00582-TYDavTf8oyy  00614-ki6Cu76pWzF  00666-GNGYKt8XrjF  00707-XVSZJAtHKdi  00743-31DHHWieDMS  00760-5Poh4Qz68hd  00790-qQgcM8T4hiD</pre>

**interiorgs** (2)

<pre>interior_0654_841467  interior_0656_841470</pre>

**mp3d** (28)

<pre>1LXtFkjw3qL  1pXnuDYAj8r  29hnd4uzFmX  5ZKStnWn8Zo  82sE5b5pLXE  ARNzJeq3xxb  D7N2EKCX4Sj  EDJbREhghzL  EU6Fwq7SyZv  SN83YJsR3w2  TbHJrupSAjP  VFuaQ6m2Qom  VVfe2KiqLaN  Vvot9Ly1tCj  VzqfbhrpDEA  X7HyMhZNoso  XcA2TqTSSAj  jtcxE69GiFV  mJXqzFtmKg4  pRbA3pwrgk9  r1Q1Z4BcV1o  rPc6DW4iMge  rqfALeAoiTq  s8pcmisQ38h  sKLMLpTHeUy  sT4fr6TAbpF  uNb9QFRL6hY  yqstnuAEVhm</pre>


## Commercial

supermarkets, convenience stores, restaurants, malls, entertainment venues, hotels and guesthouses. 10 scenes, 195 episodes.

**habitat_gs** (1)

<pre>scene64</pre>

**hm3d** (2)

<pre>00117-2NwLiyeKcrK  00403-t3t9ofFLcFU</pre>

**interiorgs** (2)

<pre>interior_0409_840098  interior_0792_839972</pre>

**mp3d** (5)

<pre>17DRP5sb8fy  HxpKQynjfin  PuKPg4mmafe  r47D5H71a5s  x8F5xyUWy9e</pre>


## Institution

schools, offices, hospitals, libraries, churches, exhibition halls, historic buildings. 11 scenes, 219 episodes.

**hm3d** (1)

<pre>00429-UZ5rYGiwQgW</pre>

**interiorgs** (3)

<pre>interior_0399_840130  interior_0405_840145  interior_0791_840126</pre>

**mp3d** (7)

<pre>8194nk5LbLH  B6ByNegPMKs  VLzqgDo317F  YVUC4YcDtcY  Z6MFQCViBuw  ZMojNkEp431  gYvKGZ5eRqb</pre>


## Outdoor

everything outdoors: streets, open ground and campuses. 8 scenes, 228 episodes.

**habitat_gs** (8)

<pre>scene56  scene57  scene58  scene59  scene60  scene61  scene62  scene65</pre>
