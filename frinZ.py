#!/usr/bin/env python3
# AKIMOTO
HISTORY = \
""" **********************
 History
  2022/06/10
  2022/06/11 update1
  2022/06/12 update2
  2022/06/14 update3
  2022/06/16 update4
  2022/06/17 update5
  2022/06/19 update6
  2022/06/20 update7
  2022/06/21 update8
  2022/07/09 update9
  2022/07/10 update10
  2022/07/10 update11
  2022/07/11 update12
  2022/07/12 update13
  2022/07/13 update14
  2022/07/15 update15
  2022/07/18 update16
  2022/07/22 update17
  2022/07/23 update18
  2022/08/22 update19
  2022/09/28 update20
  2023/01/17 update21
  2023/01/24 update22
  2023/02/28 update23
  2023/04/03 update24
  2023/05/09 update25
  2023/05/16 update26
  2023/09/23 update27
  2023/11/26 update28
  2024/01/21 update29
  2024/01/24 update30
  2024/03/08 update31
  2024/07/17 update32
  2024/11/20 update33
  2024/11/30 update34
  2024/12/29 update35
  2025/05/10 update36
  2025/05/31 update37
  
  Made by AKIMOTO on 2022/06/10
********************** """

VERSION = \
"""+++ Version +++
 version 1.0 (2022/07/12): とりあえず完成した（S/N の評価はまだ）．
 version 1.1 (2022/07/13): numba (jit) を使用しなくてもいいようにした．
 version 1.2 (2022/07/15): 積分時間を累積することにより，積分時間 vs S/N のグラフを出力できるようにした．
 version 1.3 (2022/07/18): フリンジの３次元グラフを html で保存してグラフを動かせるようにした．
 version 1.4 (2022/07/22): 32m & 34m が観測する天体の方向を az-el で出力できるようにした．
 version 1.5 (2022/07/22): 解析したデータの周波数をファイル名に組み込んだ．
 version 1.6 (2022/08/22): Delay-Rate 平面の表示範囲を変更して，fringe と同じ表示範囲にして比較しやすいようにした．
 version 1.7 (2022/09/28): FFT 点数が 1024 でないとき，正しく RFI カットをすることができなかったので，修正した．
 version 1.8 (2023/01/17): version 1.7 以前から残っていた，複数範囲の RFI カットのバグを修正．これまでは --rfi 1,50 250,300 としても 250-300 MHz しかカットされなかった．ついでにシェバンも修正．
 version 1.9 (2023/01/24): 積分時間が短いときの delay-rate 平面のカラーマップの表示範囲がおかしいので修正。
 version 1.10 (2023/02/28): 積分時間を累積してフリンジ出力するときに --length と --loop を --cumulate と同時に指定しないとダメだったが，修正して --cumulate だけで可能にした．またノイズレベルが累積した積分時間に応じてルートで減少することを確かめるグラフを出力するようにした．ついでにグラフの目盛りを上下左右に表示されるようにした．
 version 1.11 (2023/04/03): 出力する画像ファイルに記述されているパラメータのフォーマットを変更した．このプログラムで出力するファイルをまとめるように出力先のディレクトリを設定した．delay-rate 平面と freq-rate 平面の図が小さいので，拡大した図を postscript で出力するようにした．これで図の拡大が容易になる．
 version 1.12 (2023/05/16): フリンジの出力に MJD と DATE の列を追加した．
 version 1.13 (2023/11/26): VLBI 観測を実施してターゲットのフラックス密度が検出限界に近いとき，NICT の fringe では天体信号を検出せず，フリンジサーチ内の最大値を取得する可能性が大いにある．VLBI データの解析では，ゲインキャリブレーターで delay と rate を決定してターゲットに適用してやれば，微弱なターゲットのフリンジでも delay と rate 付近に現れる．これを取得するために delay と rate を任意で指定してフリンジサーチの範囲を制限し，その範囲の最大値を取得する．
 version 1.14 (2024/01/21): time domain のフリンジのカラーマップを表示範囲内で色付けする．混信が天体信号に対して十分に強いと，カラーマップ上で天体のフリンジがノイズに埋もれているように見えてしまう．これを回避するために追加した．
 version 2.0 (2024/01/24): cor ファイルを読み込むプログラムを変更した．
 version 3.0 (2024/03/08): FFT/IFFT をマルチプロセスで動くように変更した（あんまり意味はなかった）．
 version 3.1 (2024/07/17): namedtupe を用いてヘッダーを読み込むようにした．
 version 4.0 (2024/11/20): Delay と Rate を補正できるようになった（VLBI 技術を参考にした）．
 version 4.1 (2024/11/30): グラフの描画に RAM が大量に消費されていたので，contour --> imshow に変更した．
 version 4.2 (2024/12/29): JSON フォーマットで res-delay, res-rate, res-acel を指定できるようにした．--delay-corr, --rate-corr, and --acel-corr を使わなくてもよい．
 version 4.3 (2025/05/31): クロスパワースペクトルとフリンジの全てのデータを fits で表示できるようにした．そうすると ds9 で表示することができ，縦断面と横断面を同時に使用することで，png や pdf では発見しづらい混信やスプリアスを発見しやすくする．
 +++"""

import os
import sys
import datetime
import argparse
import numpy as np
import scipy.fft
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib
matplotlib.style.use('fast')
matplotlib.use("Agg")
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"]     = "in"       
plt.rcParams["ytick.direction"]     = "in"       
plt.rcParams["xtick.minor.visible"] = True       
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"]           = True
plt.rcParams["xtick.bottom"]        = True
plt.rcParams["ytick.left"]          = True
plt.rcParams["ytick.right"]         = True  
plt.rcParams["xtick.major.size"]    = 5          
plt.rcParams["ytick.major.size"]    = 5          
plt.rcParams["xtick.minor.size"]    = 3          
plt.rcParams["ytick.minor.size"]    = 3          
plt.rcParams["axes.grid"]           = False
plt.rcParams["grid.color"]          = "lightgray"
plt.rcParams["axes.labelsize"]      = 15
plt.rcParams["font.size"]           = 12


fname = os.path.basename(sys.argv[0]) # fname is "f"ile name
DETAIL = \
F"""
# -------------------------- #
#  プログラムの使用例を示す．#
# -------------------------- #
以下にプログラムの使用例を示す．例で使用している引数は，
個々の事例に細分化して指定しているだけなので，
それぞれの引数を組み合わせて使用することができる．

1. cor ファイルのフリンジやスペクトルの出力結果だけを見たいとき
---------------------------------------------------------------
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency

　周波数領域の結果は引数に --frequency を与えないと出力されない．

2. フリンジとやスペクトルの結果をテキストで出力したいとき
---------------------------------------------------------
　1. の例の末尾に --output を追加するだけ．ただし，オプションの順番は問わない．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --output
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --output

3. フリンジとスペクトルのグラフを出力したいとき
-----------------------------------------------
　こちらも 2. と同様に 1. の末尾に --plot を追加するだけ．
　併せてテキストにも出力したいなら，--output も追加する．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --plot
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --plot

　次から説明する --length や --loop を併せて用いることで，
　指定した積分時間ごとでもグラフを出力できる．

4. 積分時間を変えてフリンジやスペクトルを出力したいとき
-------------------------------------------------------
　--length を引数に追加する．例として積分時間１０秒で出力したいとき
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10
    
　しかし，これでは観測開始時刻から１０秒のフリンジやスペクトルの出力しかしない．
　観測時間を --length で指定した積分時間で区切って出力したいときは，次の --loop を追加する．

5. 指定した積分時間ごとにフリンジやスペクトルの出力を見たいとき
---------------------------------------------------------------
　4. の例では，cor ファイルに記述されている観測開始時間から１０秒分の
　フリンジやスペクトルしか見れない．指定した積分時間でフリンジやスペクトル
　を出力したいときは，4. の例に --loop を追加する．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10 --loop 20
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10 --loop 20

　上記の例では，積分時間１０秒でフリンジやスペクトルを計算し，それを２０回
　繰り返す，つまり観測開始時刻から２００秒間のデータを１０秒間隔でそれらを
　出力する．

6. フリンジやスペクトルを出力する開始時刻をずらしたいとき
---------------------------------------------------------
　4. や 5. では，cor ファイルに記述されている観測開始時刻から指定した積分時間分
　だけの出力を行っている．観測開始時刻から１７秒だけずらしてからフリンジやスペクトル
　を出力したいときは --skip を追加する．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10 --loop 20 --skip 17
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10 --loop 20 --skip 17

　2. でも述べているが，--length や --loop，--skip の順序は問わない．しかし，
　それぞれに指定する値が異なると，違った出力になる．

7. RFI カットをしたいとき
-------------------------
　こちらも 2. と同様に 1. の末尾に --rfi を追加するだけ．
　RFI をカットしたい周波数の範囲を 1-150 MHz とすると
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --rfi 1,150
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --rfi 1,150

　他にも 300-450 MHz もカットしたいときは
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --rfi 1,150 300,450
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --rfi 1,150 300,450
    
　1,150 と 300,450 は順不同である．

8. 積分時間を累積して，積分時間に対する S/N のグラフを出力したいとき
------------------------------------------------------------------
　引数 --cumulate を指定する
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --cumulate 1

　とすると，積分時間１秒で１秒ごと（１秒，２秒，３秒...）にフリンジを計算する．

9. ダイナミックスペクトルを出力したいとき
-----------------------------------------
　引数に --dynamic-spectrum を追加する．ダイナミックスペクトルの出力では --length や --loop，--skip を
　指定しても無視される．また --frequency の有無にも関わらず出力する．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --dynamic-spectrum
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --dynamic-spectrum

　上記のどちらにしても同じダイナミックスペクトルを得ることができる．

10. Delay-Rate サーチ平面の３次元プロットを出力したいとき
--------------------------------------------------------
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --3D

11. Frequency-Rate の２次元データ，Delay-Rate の２次元データを出力したいとき
----------------------------------------------------------------------------
　引数に --cross-output を指定する．
    Time: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --cross-output
    Freq: {fname} --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --cross-output

以上（2022/07/11 AKIMOTO）""" 

DESCRIPTION = \
F"""
# ---------- #
# 簡単な説明 #
# ---------- #
　{fname} は fringe コマンドの補助となるプログラムである．fringe は Delay-Rate サーチ平面でピーク値を持っているが，
　これでは混信が見られるデータでは，こちらを検出してしまう．山口干渉計では，キャリブレーターを用いて，きちんと
　遅延時間を決定すれば，混信の影響を受けずに (Delay, Rate) = (0, 0) に天体の信号が現れる（検出できれば...）．
　このプログラムでは，サーチ内のピークではなく，(Delay, Rate) = (0, 0) の信号を必ず取ってくる．こうすれば，
　混信の影響を受けずに天体信号を取り出すことができるので，fringe コマンドで行っていた RFI のカットをしなくて
　よくなる．しかし，(Delay, Rate) = (0, 0) でも混信が見られるなら RFI カットの必要はある．

# ---------- #
# 引数の詳細 #
# ---------- #
　{fname} の詳細な使い方は \"{fname} --detail\" を実行することで確認できる．
""" 

EPILOG = \
"""\n
# ---------- #
# 修正や追加 #
# ---------- #
　プログラムのバグや処理の追加などがありましたら，穐本（c001wbw[at]yamaguchi-u.ac.jp，[at] を @ へ変更してください）まで
　連絡してください．連絡先は 2022/07/11 時点のものです．
"""
# arguments
class MyHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG, formatter_class=MyHelpFormatter)  

help_input     = "cor ファイルを指定する（１つまで）．"
help_length    = "積分時間（PP単位，基本的には秒単位）を指定する．"
help_skip      = "cor ファイル中の観測開始時刻から，どれくらいの時間を飛ばすか．"
help_loop      = "引数 --length で指定した積分時間分のフリンジ出力を何回繰り返すか．"
help_output    = "フリンジもしくはスペクトルの出力をテキストファイルとして出力する．"
help_plot      = "フリンジのグラフを出力する．スペクトルのグラフを出力したいときは，引数 --frequency と同時に使用する．"
help_freq      = "スペクトルのグラフを出力する．引数 --plot と同時に使用する．"
help_rfi       = "周波数カット．切り取りたい周波数の最小値と最大値を次のように指定する．--rfi freq_min,freq_max．複数の周波数カットをしたい場合は --rfi freq_min1,freq_max1 freq_min2,freq_max2 ... とする．"
help_cumulate  = "指定した時間間隔で積分時間を累積していき，積分時間 vs S/N の両対数グラフを出力する．また積分時間 vs ノイズレベルのグラフも出力する．"
help_cpu       = "FFT/IFFT を実行する際に使用する CPU の数（ビジビリティが多い場合だけおそらく速くなる）"
help_cross     = "Frequency-Rate の２次元データ，Delay-Rate の２次元データを csv ファイルとして出力する．同時に cor ファイルのデータも出力する．"
help_dynamic   = "ダイナミックスペクトルとフリンジの時間変化を出力する．混信の把握に便利．"
help_3D        = "Delay-Rate サーチ平面の３次元プロットを表示する．２パターンの画像が出力され，１つはすべてのサーチ平面をプロットしており，１つは (Delay,Rate) = (0,0) の部分を拡大したサーチ平面をプロットしている．"
help_add_plot  = "相関振幅，位相，S/N の時間変化をプロットする．"
help_plane     = "--plot で出力される delay-rate 平面や frequency-rate 平面の拡大図を出力する．--plot と組み合わせて用いる．"
help_fits      = "--plot と --plane-expansion と同時に使用する．delay-rate 平面や frequency-rate 平面の全体のデータを fits に出力する．ds9 で編集できる．"
help_del_win   = "--delay-window delay-range-min delay-range-max．VLBI 観測を実施してターゲットのフラックス密度が検出限界に近いとき，NICT の fringe では天体信号を検出せず，フリンジサーチ内の最大値を取得する可能性が大いにある．VLBI データの解析では，ゲインキャリブレーターで delay と rate を決定してターゲットに適用してやれば，微弱なターゲットのフリンジでも delay と rate 付近に現れる．これを取得するために delay と rate を任意で指定してフリンジサーチの範囲を制限し，その範囲の最大値を取得する．"
help_rate_win  = "--delay-window の rate 版"
help_del_corr  = "Delay (sample 単位) を補正する．fringe の出力もしくは frinZ.py を複数回実行して特定した値をそのまま引数にとる．"
help_rate_corr = "Rate (Hz 単位) を補正する．fringe の出力もしくは frinZ.py を複数回実行して特定した値をそのまま引数にとる．"
help_acel_corr = "test"
help_delay_rate_acel_json = "JSON フォーマットで res-delay と res-rate をまとめたファイルを指定する．JSON ファイルの中身は  \"cor-file path\": [res-delay,res-rate] とする．"
help_serach    = "--peak-search Delay Rate．Delay は sample 単位で Rate は Hz 単位．その周囲でフリンジのピークを探査して，フリンジのピークの詳細な Delay と Rate を決定する．"
help_cmap_del  = "time domain のフリンジのカラーマップを表示範囲内で色付けする．混信が天体信号に対して十分に強いと，カラーマップ上で天体のフリンジがノイズに埋もれているように見えてしまう．これを回避するために追加した．"
help_summarize = "すべての出力ファイルをディレクトリにまとめる．"
help_history   = "このプログラムの変更履歴を表示する．"
help_version   = "このプログラムのバージョンを表示する．"
help_detail    = "このプログラムの使い方の詳細を表示する．"
help_header    = "cor ファイルのヘッダーを表示する．"

parser.add_argument("--input"           , default=True , type=str              , help=help_input   )
parser.add_argument("--length"          , default=0    , type=int              , help=help_length  )
parser.add_argument("--skip"            , default=0    , type=int              , help=help_skip    )
parser.add_argument("--loop"            , default=False, type=int              , help=help_loop    )
parser.add_argument("--output"          , action="store_true"                  , help=help_output  )
parser.add_argument("--plot"            , action="store_true"                  , help=help_plot    )
parser.add_argument("--frequency"       , action="store_true"                  , help=help_freq    )
parser.add_argument("--rfi"             , default=False, nargs="*"             , help=help_rfi     )
parser.add_argument("--delay-window"    , default=False, nargs=2 , dest="del_win", help=help_del_win )
parser.add_argument("--rate-window"     , default=False, nargs=2 , dest="rate_win", help=help_rate_win)
parser.add_argument("--delay-correct"   , default=0.0, type=float, dest="del_corr", help=help_del_corr)
parser.add_argument("--rate-correct"    , default=0.0, type=float, dest="rate_corr", help=help_rate_corr)
parser.add_argument("--acel-correct"    , default=0.0, type=float, dest="acel_corr", help=help_acel_corr)
parser.add_argument("--delay-rate-acel-json" , default=False, dest="delay_rate_acel_json", help=help_delay_rate_acel_json)
parser.add_argument("--peak-search"     , default=False, nargs=2   , dest="peak_search", help=help_serach)
parser.add_argument("--cumulate"        , default=0    , type=int              , help=help_cumulate)
parser.add_argument("--cpu"             , default=1    , type=int              , help=help_cpu     )
parser.add_argument("--cmap-time"       , action="store_true", dest="cmap_time", help=help_rate_win)
parser.add_argument("--cross-output"    , action="store_true", dest="cross"    , help=help_cross   )
parser.add_argument("--dynamic-spectrum", action="store_true", dest="dynamic"  , help=help_dynamic )
parser.add_argument("--3D"              , action="store_true", dest="ddd"      , help=help_3D      )
parser.add_argument("--add-plot"        , action="store_true", dest="addplot"  , help=help_add_plot)
parser.add_argument("--plane-expansion" , action="store_true", dest="plane"    , help=help_plane   )
parser.add_argument("--fits"            , action="store_true"                  , help=help_fits    )
parser.add_argument("--history"         , action="store_true"                  , help=help_history )
parser.add_argument("--version"         , action="store_true"                  , help=help_version )
parser.add_argument("--detail"          , action="store_true"                  , help=help_detail  )
parser.add_argument("--header"          , action="store_true", default=False    , help=help_header  )

args = parser.parse_args() 
ifile     = args.input
length    = args.length
skip      = args.skip
loop      = args.loop
output    = args.output
time_plot = args.plot
freq_plot = args.frequency
rfi       = args.rfi
cpu       = args.cpu
cumulate  = args.cumulate
cs_output = args.cross # cs: "c"ross-"s"pectrum
add_plot  = args.addplot
delay_win = args.del_win
rate_win  = args.rate_win
delay_correct = args.del_corr
rate_correct = args.rate_corr
acel_correct = args.acel_corr
delay_rate_acel_json = args.delay_rate_acel_json
peak_search = args.peak_search
cmap_time = args.cmap_time
DDD       = args.ddd   # 2D-graph
ds_plot   = args.dynamic
plane     = args.plane
fits      = args.fits
history   = args.history
version   = args.version
detail    = args.detail
header_   = args.header


def help_show() :
    os.system("%s --help" % os.path.basename(sys.argv[0]))
    exit(0)
if len(sys.argv) == 1 or not ifile :
    help_show()
if history == True :
    print(HISTORY) ; exit(0)
if detail == True  :
    print(DETAIL)  ; exit(0)
if version == True :
    print(VERSION) ; exit(0)


def Radian2RaDec(RA_radian: float, Dec_radian: float) -> float :
    Ra_deg  = np.rad2deg(RA_radian)
    Dec_deg = np.rad2deg(Dec_radian) 
    return Ra_deg, Dec_deg

def RaDec2AltAz(object_ra: float, object_dec: float, observation_time: float, latitude: float, longitude: float, height: float) -> float :
    location_geocentrice = EarthLocation.from_geocentric(latitude, longitude, height, unit=u.m)
    location_geodetic    = EarthLocation.to_geodetic(location_geocentrice)
    location_lon_lat     = EarthLocation(lon=location_geodetic.lon, lat=location_geodetic.lat, height=location_geodetic.height)
    obstime              = Time(F"{observation_time}")
    object_ra_dec        = SkyCoord(ra=object_ra*u.deg, dec=object_dec*u.deg)
    AltAz_coord          = AltAz(location=location_lon_lat, obstime=obstime)
    object_altaz         = object_ra_dec.transform_to(AltAz_coord)
    
    return object_altaz.az.deg, object_altaz.alt.deg, location_lon_lat.height.value

def zerofill(integ: int) -> int :
    powers_of_two = 1
    integ_re      = integ
    while True :
        powers_of_two = powers_of_two * 2
        integ = integ / 2
        if integ < 1.0 :
            break
        else :
            continue
    zero_num = int(powers_of_two - integ_re)
    return powers_of_two, zero_num

def RFI(r0, bw: int, fft: int) -> int :
    # RFI
    rfi_cut_min = []
    rfi_cut_max = []
    for r1 in r0 :
        rfi_range = r1.split(",")
        rfi_min = int(rfi_range[0])
        rfi_max = int(rfi_range[1])
        if rfi_max > 512 :
            rfi_max = 512
        if rfi_min < 0 or rfi_max < 0 :
            print("The RFI minimum, %.0f, or maximum, %.0f, frequency is more than 0." % (rfi_min, rfi_max))
            exit(1)
        elif rfi_min >= rfi_max :
            print("The RFI maximum frequency, %.0f, is smaller than the RFI minimum frequency, %.0f." % (rfi_min, rfi_max))
            exit(1)
        else :
            pass

        r2 = int(rfi_min) * int(fft/2/bw); rfi_cut_min.append(r2)
        r3 = int(rfi_max) * int(fft/2/bw); rfi_cut_max.append(r3)
        
    return rfi_cut_min, rfi_cut_max, len(r0)

#from numba import jit
#@jit
def noise_level(input_2D_data: float, search_00_amp: float) -> float :
    
    input_2D_data_real_imag_ave = np.mean(input_2D_data) # 複素数でも実部と虚部でそれぞれで平均を計算できるみたい．
    noise_level = np.mean(np.absolute(input_2D_data - input_2D_data_real_imag_ave)) # 信号の平均値が直流成分に対応するため，それを除去のために平均値を引いている？．加算平均をとることで雑音レベルを下げることができるらしい．
    
    try :
        SNR = search_00_amp / noise_level
    except ZeroDivisionError :
        SNR, noise_level = 0.0, 0.0

    return SNR, noise_level

def time_add_plot(target, x1, y1, xl, yl, s, Xm, xM, save_pass) :
    fig = plt.figure(figsize=(9,6))
    plt.plot(x1, y1, "o", label="%s\nlength: %d s" % (target, length))
    plt.xlabel("The elapsed time since %s UT" % xl)
    plt.ylabel("%s" % yl)
    plt.xlim([Xm, xM])
    if s == "snr" :
        plt.ylim(ymin=0)
    elif s == "phase" :
        plt.yticks([-180, -120, -60, 0, 60, 120, 180])
        plt.ylim(-180, 180)
    plt.tight_layout()
    plt.legend()
    plt.savefig("%s_%s.png" % (save_pass, s))
    #plt.show()
    plt.clf(); plt.close()


def dynamic_spectrum(x,y,z,xl,yl,zl,xr,yr,path) :
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), tight_layout=True)
    #c = ax.contourf(x, y, np.absolute(z), 100, cmap="rainbow")
    c = ax.imshow(np.absolute(z), extent=[x[0], x[-1], y[-1], y[0]], aspect="auto", cmap="rainbow")
    fig.colorbar(c, label=zl)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xlim(xr)
    plt.ylim(yr)
    #plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.clf(); plt.close()

def digit_cal(num) :
    if type(num) == int :
        digit = int(len(str(num)))
    elif type(num) == float :
        pass
    else :
        print("Please specify an inter or a float number.")
        exit()
    return digit

# label
label = os.path.splitext(ifile)[0].split("_")[-1]

#
# the directory to save the output data
#
save_directory_path = os.path.dirname(ifile)
if save_directory_path == "" :
    save_directory_path = "./frinZ"
else :
    save_directory_path += "/frinZ"
os.makedirs(save_directory_path, exist_ok=True)

#
# header region
#
header_save_path = save_directory_path + "/cor_header"
os.makedirs(header_save_path, exist_ok=True)
cor_header_file = "%s/%s_header.txt" % (header_save_path, os.path.basename(ifile).split(".")[0])
cor_header_file_exist = os.path.isfile(cor_header_file)
    
header1 = np.fromfile(ifile, dtype="<i4", count=8).tolist()  # SoftWare Version, Sampling Freq, FFT point, Number of Sector
header2 = np.fromfile(ifile, dtype="<f8", count=32).tolist() # Station 1&2 XYZ position, Sorce Position, Clock Delay
header3 = np.fromfile(ifile, dtype="<S8", count=17, offset=32).tolist() # Station Name, Station Code, Source Name
header_field = ["version", "software", "sampling_speed", "fft", "sector", "frequency", 
                "station1_posx", "station1_posy", "station1_posz", 
                "station2_posx", "station2_posy", "station2_posz", 
                "source_ra", "source_dec", 
                "station1_delay", "station1_rate", "station1_acel", "station1_jerk", "station1_snap", 
                "station2_delay", "station2_rate", "station2_acel", "station2_jerk", "station2_snap", 
                "station1_name" , "station1_code", "station2_name", "station2_code", "source_name"]
header = namedtuple("Header", header_field)
header = header(header1[1], header1[2], header1[3], header1[6], header1[7], 
                header2[2], header2[6], header2[7], header2[8], header2[12], header2[13], header2[14], header2[18], header2[19], 
                header2[21], header2[22], header2[23], header2[24], header2[25], header2[27], header2[28], header2[29], header2[30], header2[31], 
                header3[0].decode(), header3[5].decode(), header3[6].decode(), header3[11].decode(), header3[12].decode())

magic_word          = "3ea2f983"
header_version      = header.version
software_version    = header.software
sampling_speed      = header.sampling_speed / 10**6

# FFT point, Number of Sector, observing frequency and parametr period
fft_point           = header.fft
number_of_sector    = header.sector
observing_frequency = header.frequency/ 10**6
PP  = number_of_sector         # parameter period
BW  = int(sampling_speed // 2) # 
RBW = BW / (fft_point // 2)    # resolution bandwidth

# Station1
station1_name       = header.station1_name
station1_position_x = header.station1_posx
station1_position_y = header.station1_posy
station1_position_z = header.station1_posz
station1_code       = header.station1_code

# Station2
station2_name       = header.station2_name
station2_position_x = header.station2_posx
station2_position_y = header.station2_posy
station2_position_z = header.station2_posz
station2_code       = header.station2_code

# Source-Name
source_name         = header.source_name
source_position_ra, source_position_dec = Radian2RaDec(header.source_ra, header.source_dec)

station1_clock_delay = header.station1_delay
station1_clock_rate  = header.station1_rate
station1_clock_acel  = header.station1_acel
station1_clock_jerk  = header.station1_jerk
station1_clock_snap  = header.station1_snap
station2_clock_delay = header.station2_delay
station2_clock_rate  = header.station2_rate
station2_clock_acel  = header.station2_acel
station2_clock_jerk  = header.station2_jerk
station2_clock_snap  = header.station2_snap

#       
# Header Region Information
#
header_region_info = \
f"""##### Header Region
Magic Word = {magic_word}
Sofrware Vesion = {software_version:.0f}
Header Version = {header_version:.0f}
Sampling frequency = {sampling_speed:.0f} MHz
Observing Frequency = {observing_frequency:.0f} MHz
FFT Point = {fft_point:.0f}
PP (parameter period) = {number_of_sector:.0f}
BandWidth = {BW:.0f} MHz
Resolution BandWidth = {RBW:.5f} MHz

Station1
    Name = {station1_name}
    Code = {station1_code}
    Station1 Clock Delay = {station1_clock_delay} s
    Station1 Clock Rate  = {station1_clock_rate} s/s
    Station1 Clock Acel  = {station1_clock_acel} s/s**2
    Station1 Clock Jerk  = {station1_clock_jerk} s/s**3
    Station1 Clock Snap  = {station1_clock_snap} s/s**4
    Pisition (X,Y,Z) = ({station1_position_x},{station1_position_y},{station1_position_z}) m, geocentric coordinate

Station2
    Name = {station2_name}
    Code = {station2_code}
    Station2 Clock Delay = {station2_clock_delay} s
    Station2 Clock Rate  = {station2_clock_rate} s/s
    Station2 Clock Acel  = {station2_clock_acel} s/s**2
    Station2 Clock Jerk  = {station2_clock_jerk} s/s**3
    Station2 Clock Snap  = {station2_clock_snap} s/s**4
    Pisition (X,Y,Z) = ({station2_position_x},{station2_position_y},{station2_position_z}) m, geocentric coordinate

Source
    Name = {source_name}
    Position (RA, Decl) = ({source_position_ra:.5f},{source_position_dec:.5f}) deg, J2000
""" 
cor_header_save = open(cor_header_file, "w"); cor_header_save.write(header_region_info); cor_header_save.close()
if  header_ : print("Header Show\n", header_region_info)


#
# adjust the length, the skip and the integration time
#

# length
if length <= 0 or length > PP :
    length = PP
if length == PP :
    length_label = "0"
else :
    length_label = "%d" % length

# skip
if skip < 0 or skip >= PP:
    print("As following the condition, please specify the skip argument.")
    print("# 0 < skip <= %f" % PP)
    print("You specify the skip time, %.0f" % skip)
    exit(1)


# loop
if loop == False or (PP-skip)//length <= 0  :
    loop = 1
elif loop >= (PP-skip)//length :
    loop = (PP-skip)//length

# cummulate
cumulate_output = ""
if cumulate >= PP :
    print("The specified cumulation length, %d s, is more than the observation time, %d s." % (cumulate, PP))
    exit(1)
if cumulate != 0 and (add_plot == True or freq_plot == True) :
    print("You can specify the argument whether \"--cumulate\" or \"--add-plot\"")
    print("and \"--cumulate\" can't specify with \"--frequency\".")
    exit(1)
if cumulate != 0 :
    length = 0
    loop = int(PP/cumulate)

# 
if plane == True and time_plot == False :
    print("The \"--plane-expansion\" option is used with the \"--plot\" option")
    exit(1)


# frequency
if   6600 <= observing_frequency <= 7112 :
    observing_band = "c"
elif 8192 <= observing_frequency <= 8704 :
    observing_band = "x"
else :
    observing_band = ""


#
# the range of the delay & rate window
#
if (delay_win != False and rate_win == False) or (delay_win == False and rate_win != False):
    print("if you specify the search window of fringe in the delay-rate plane")
    print("you must specify both --delay-window and --rate-window.")
    exit()
if delay_win != False :
    delay_window_low, delay_window_high = delay_win
    delay_win_label = "%s-%s" % (delay_window_low, delay_window_high)
    if float(delay_window_low) > float(delay_window_high) :
        print("you mistake the range of the delay window (high: %s, low: %s)." % (delay_window_low, delay_window_high))
        exit()
if rate_win != False :
    rate_window_low, rate_window_high = rate_win
    rate_win_label = "%s-%s" % (rate_window_low, rate_window_high)
    if float(rate_window_low) > float(rate_window_high) :
        print("you mistake the range of the rate window (high: %s, low: %s)." % (rate_window_low, rate_window_high))
        exit()
        

# CPU
cpu_check = os.cpu_count()
if cpu > cpu_check :
    cpu = cpu_check
else :
    pass



#
# load corss-spectrum from the inputted cor-file
#
cor_file = open(ifile, "rb")
complex_visibility = np.frombuffer(cor_file.read(), dtype="<f4", offset=256)
complex_visibility = complex_visibility.reshape(PP, int(len(complex_visibility)/PP))
effective_integration_length = complex_visibility[:,28][0]
complex_visibility = np.delete(complex_visibility, np.linspace(0,33,34, dtype=int), 1)
complex_visibility = np.insert(complex_visibility,0,0,axis=1)
complex_visibility = np.insert(complex_visibility,1,0,axis=1)
complex_visibility = complex_visibility.reshape(int(PP*(fft_point)/2), 2)
complex_visibility = (complex_visibility[:,0] + complex_visibility[:,1] *1j).reshape(PP, int(fft_point/2))
complex_visibility = complex_visibility[skip:]
cor_file.close()


digit = 0
if np.round(effective_integration_length, 0) < 1.0 :
    while effective_integration_length < 1.0 :
        effective_integration_length *= 10
        digit += 1
    effective_integration_length = 1/10**(digit -1)
else :
    effective_integration_length = np.round(effective_integration_length, 0)

cor_file = open(ifile, "rb")
obs_scan_time = np.frombuffer(cor_file.read(), dtype="<i4", offset=256)
obs_scan_time = obs_scan_time.reshape(PP, int(len(obs_scan_time)/PP))[:,0].tolist()
obs_scan_time = obs_scan_time[skip:]
cor_file.close()


aspect = int(fft_point/2)/PP
cross_spectrum_directory = save_directory_path + "/raw_visibility"
os.makedirs(cross_spectrum_directory, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.5,6))
plt.imshow(np.angle(complex_visibility, deg=True), extent=[0, fft_point/2, PP, 0],cmap='jet',aspect=aspect) 
plt.colorbar(label="The phase of the cross-spectrum (deg)")
plt.xlabel("Channel \n(Obs. Freq: %d MHz, BW: %d MHz)" % (observing_frequency, BW))
plt.ylabel("PP")
plt.tight_layout()
plt.savefig("%s/%s_raw_vis_pp_bw_phase.png" % (cross_spectrum_directory, os.path.basename(ifile).split(".")[0]))
plt.cla(); plt.close()

fig, ax = plt.subplots(figsize=(7.5,6))
plt.imshow(np.abs(complex_visibility), extent=[0, fft_point/2, PP, 0],cmap='jet',aspect=aspect) 
plt.colorbar(label="The amplitude of the cross-spectrum")
plt.xlabel("Channel \n(Obs. Freq: %d MHz, BW: %d MHz)" % (observing_frequency, BW))
plt.ylabel("PP")
plt.tight_layout()
plt.savefig("%s/%s_raw_vis_pp_bw_amp.png" % (cross_spectrum_directory, os.path.basename(ifile).split(".")[0]))
plt.cla(); plt.close()


#
# corrections of delay and rate
#
if delay_correct != 0.0 and rate_correct != 0.0 and delay_rate_acel_json != False :
    print("Please select --delay-corr and --rate-corr or --delay-rate-json to correct res-delay and res-rate.")
    exit()
if delay_rate_acel_json :
    import json
    json_load = json.load(open(delay_rate_acel_json, "r"))
    try :
        delay_correct, rate_correct, acel_correct = json_load[ifile]
    except KeyError :
        print("# Not found res-delay and res-rate of %s so they are forced to be 0.0." % ifile)
        print("# Plese use --delay-corr, --rate-corr, and --acel-corr options!")
        delay_correct, rate_correct, acel_correct = 0.0, 0.0, 0.0 
PP_correct = np.array([np.linspace(skip+1,PP,PP-skip, dtype=int)]).T
BW_correct = np.linspace(0, int(sampling_speed/2) -1, int(fft_point/2)) *10**6 # MHz
RF_correct = np.meshgrid(BW_correct, PP_correct.T)[0] + observing_frequency*10**6  # MHz
complex_visibility *= np.exp(-2*np.pi*1j*delay_correct/(sampling_speed*10**6)*BW_correct) * np.exp(-2*np.pi*1j*rate_correct*(PP_correct*effective_integration_length)) #* (1/2 * np.exp(-2*np.pi*1j*acel_correct*(PP_correct*effective_integration_length)**2))



if delay_correct != 0 or rate_correct != 0 :
    fig, ax = plt.subplots(figsize=(7.5,6))
    plt.imshow(np.angle(complex_visibility , deg=True), extent=[0, fft_point/2, PP, 0],cmap='jet',aspect=aspect) 
    plt.colorbar(label="The corrected phase of the cross-spectrum (deg)")
    plt.xlabel("Channel \n(Obs. Freq: %d [MHz], BW: %d [MHz])" % (observing_frequency, BW))
    plt.ylabel("PP")
    plt.tight_layout()
    plt.savefig("%s/%s_corrected_vis_pp_bw_phase.png" % (cross_spectrum_directory, os.path.basename(ifile).split(".")[0]))
    plt.cla(); plt.close()

    fig, ax = plt.subplots(figsize=(7.5,6))
    plt.imshow(np.abs(complex_visibility), extent=[0, fft_point/2, PP, 0],cmap='jet',aspect=aspect) 
    plt.colorbar(label="The corrected amplitude of the cross-spectrum (deg)")
    plt.xlabel("Channel \n(Obs. Freq: %d [MHz], BW: %d [MHz])" % (observing_frequency, BW))
    plt.ylabel("PP")
    plt.tight_layout()
    plt.savefig("%s/%s_corrected_vis_pp_bw_amp.png" % (cross_spectrum_directory, os.path.basename(ifile).split(".")[0]))
    plt.cla(); plt.close()


cumulate_len, cumulate_snr, cumulate_noise = [], [], []
add_plot_length, add_plot_amp, add_plot_snr, add_plot_phase, add_plot_noise_level = [], [], [], [], []
for l in range(loop) :

    #
    # A directories to summarize an analysed data
    #
    if time_plot :
        fringe_time_freq_plot_path = save_directory_path + "/fringe_graph"
        os.makedirs(fringe_time_freq_plot_path, exist_ok=True)
    if output :
        fringe_time_freq_output_path = save_directory_path + "/fringe_output"
        os.makedirs(fringe_time_freq_output_path, exist_ok=True)
    if cumulate != 0 :
        cumulate_path = save_directory_path + "/cumulate/len%ds" % cumulate
        os.makedirs(cumulate_path, exist_ok=True)
    if add_plot and cumulate == 0 :
        add_plot_path = save_directory_path + "/add_plot"
        os.makedirs(add_plot_path, exist_ok=True)
    if ds_plot :
        dynamic_spectrum_path = save_directory_path + "/dynamic_spectrum"
        os.makedirs(dynamic_spectrum_path, exist_ok=True)
    if DDD :
        ddd_path = save_directory_path + "/3D_frigne"
        os.makedirs(ddd_path, exist_ok=True)
    if plane :
        plane_path = save_directory_path + "/search_plane_expansion"
        os.makedirs(plane_path, exist_ok=True)
    if rfi :
        rfi_history = save_directory_path + "/rfi_history"
        os.makedirs(rfi_history, exist_ok=True)
    if fits :
        fits_path = save_directory_path + "/fits_freq_delay"
        os.makedirs(fits_path, exist_ok=True)

         
    save_file_name = ""

    # the cumulation of the integration time.
    if cumulate != 0 :
        if length <= PP :
            length += cumulate
            l = 0
    if length > PP : # for --cumulate
        break

    # epoch
    epoch0 = obs_scan_time[length*l:length*(l+1)]
    epoch0 = datetime.datetime(1970,1,1,0,0,0) + datetime.timedelta(seconds=epoch0[0])
    epoch1 = epoch0.strftime("%Y/%j %H:%M:%S")
    epoch2 = epoch0.strftime("%Y%j%H%M%S")
    epoch3 = epoch0.strftime("%Y-%m-%d %H:%M:%S")


    mjd = "%.5f" % Time("T".join(epoch3.split()), format="isot", scale="utc").mjd

    # azel 
    station1_azel = RaDec2AltAz(source_position_ra, source_position_dec, epoch3, station1_position_x, station1_position_y, station1_position_z)
    station2_azel = RaDec2AltAz(source_position_ra, source_position_dec, epoch3, station2_position_x, station2_position_y, station2_position_z)

    # save-file name
    save_file_name += "%s_%s_%s_%s_%s_len%.0fs" % (station1_name, station2_name, epoch2, label, observing_band, length)
    if rfi != False:
        save_file_name += "_rfi"
    if cumulate != 0 :
        save_file_name += "_cumulate%ds" % cumulate

    save_file_path = save_directory_path

    #
    # for caribrating a delay and rate 
    #
    #PP_correct = np.array([np.linspace(length*l+1,length*(l+1), length, dtype=int)]).T
    #BW_correct = np.linspace(0, int(sampling_speed/2) -1, int(fft_point/2)) *10**6 # MHz
    #RF_correct = np.meshgrid(BW_correct, PP_correct)[0] + observing_frequency*10**6  # MHz

    
    #complex_visibility_split = complex_visibility[length*l:length*(l+1)] * np.exp(-2*np.pi*1j*delay_correct/(sampling_speed*10**6)*BW_correct)* np.array([np.exp(-2*np.pi*1j*rate_correct*RF_correct*PP_correct)]).T 
    complex_visibility_split = complex_visibility[length*l:length*(l+1)] #* np.exp(-2*np.pi*1j*delay_correct/(sampling_speed*10**6)*BW_correct) * np.exp(-2*np.pi*1j*rate_correct*PP_correct) 

    
    #
    # IFFT & FFT
    #
    integ_fft = 4 *zerofill(integ=length)[0] # the FFT in the time (same as integration time) direction. rate

    #
    # RFI cut
    #
    if rfi != False :
        rfi_cut_min, rfi_cut_max, rfi_num = RFI(r0=rfi, bw=BW, fft=fft_point)
        for i in range(rfi_num) :
            for r in range(rfi_cut_min[i], rfi_cut_max[i]+1) :
                if r >= int(fft_point/2) :
                    continue
                complex_visibility_split[:,r] = 0+0j
        
        rfi_history_txt = open(f"{rfi_history}/{save_file_name}_history.txt", "a")
        rfi_history_txt.write(f"cor: {ifile}, rfi: {rfi}, length: {length}, delay-window: {delay_win}, rate-win: {rate_win}, delay-corr: {delay_correct}, rate-corr: {rate_correct}\n")
        rfi_history_txt.close()

    # Numpy version
    #freq_rate_2D_array = np.fft.fftshift(np.fft.fft(complex_visibility_split, axis=0, n=integ_fft), axes=0) * fft_point / length
    #lag_rate_2D_array  = np.fft.ifftshift(np.fft.ifft(freq_rate_2D_array, axis=1, n=fft_point), axes=1)
    
    # Scipy version
    freq_rate_2D_array = np.fft.fftshift(scipy.fft.fft(complex_visibility_split, axis=0, n=integ_fft, workers=cpu), axes=0) * fft_point / length
    lag_rate_2D_array  = np.fft.ifftshift(scipy.fft.ifft(freq_rate_2D_array, axis=1, n=fft_point, workers=cpu), axes=1)
    lag_rate_2D_array  = lag_rate_2D_array[:, ::-1]        # 列反転，これは delay が０を中心に対称になるため．


    #
    # the cross-spectrum, the fringe phase, the rate in the frequency domain, the time-lag, and the rate in the time domain.
    #
    integ_range = np.round(np.linspace(1,int(PP*effective_integration_length),int(PP*effective_integration_length)), 5)                                             # integration time range
    rate_range  = np.fft.fftshift(np.fft.fftfreq(integ_fft, d=effective_integration_length))    # rate range, the sampling frequency is 1 second if the outout value in xml-file is 1 Hz and the parameter if length is 1.
    freq_range  = np.round(np.linspace(0,BW,fft_point//2),digit_cal(int(fft_point/2)))                                   # cross spectrum range
    lag_range   = np.linspace(-fft_point//2+1,fft_point//2,fft_point, dtype=int)        # time lag range


    #
    # YI only
    # 
    yi_time_lag  = 0.0
    yi_time_rate = 0.0
    yi_freq_rate = 0.0

    #
    # fringe search
    #
    # frequency domain
    if freq_plot == True :
        fringe_freq_rate_00_complex_index = np.where(rate_range==yi_freq_rate)[0][0]
        fringe_freq_rate_00_spectrum      = np.absolute(freq_rate_2D_array[fringe_freq_rate_00_complex_index,:])
        fringe_freq_rate_00_phase1        = np.angle(freq_rate_2D_array[fringe_freq_rate_00_complex_index,:], deg=True)
        fringe_freq_rate_00_index         = fringe_freq_rate_00_spectrum.argmax()
        fringe_freq_rate_00_amp           = fringe_freq_rate_00_spectrum[fringe_freq_rate_00_index]
        fringe_freq_rate_00_freq          = freq_range[fringe_freq_rate_00_index]
        fringe_freq_rate_00_rate          = np.absolute(freq_rate_2D_array[:,fringe_freq_rate_00_index])
        fringe_freq_rate_00_phase2        = fringe_freq_rate_00_phase1[fringe_freq_rate_00_index]
        #
        # noise level, frequency domain
        #
        SNR_freq_rate, noise_level_freq = noise_level(freq_rate_2D_array, fringe_freq_rate_00_amp)
    
    #
    # time domain
    #
    delay_win_range_low = -30
    delay_win_range_high = 30
    if (-8/length) < rate_range[0] : rate_win_range_low = rate_range[0]*effective_integration_length
    else :                           rate_win_range_low = -8/(length*effective_integration_length)
    if (8/length) < rate_range[-1] : rate_win_range_high = 8/(length*effective_integration_length)
    else :                           rate_win_range_high = rate_range[-1]*effective_integration_length

    if freq_plot != True :
        
        #
        # When the target flux density is nearly detection limit in VLBI
        #
        if delay_win != False and rate_win != False :
            delay_win_range_low = float(delay_window_low)
            delay_win_range_high = float(delay_window_high)
            rate_win_range_low = float(rate_window_low)
            rate_win_range_high = float(rate_window_high)
            delay_win_range = (float(delay_window_low) <= lag_range)  & (lag_range <= float(delay_window_high))
            rate_win_range  = (float(rate_window_low)  <= rate_range) & (rate_range <= float(rate_window_high))
            delay_rate_fringe_search_area = lag_rate_2D_array[rate_win_range][:,delay_win_range]
            delay_win_range_max_idx, rate_win_range_max_idx = np.unravel_index(np.argmax(np.absolute(delay_rate_fringe_search_area)), delay_rate_fringe_search_area.shape)
            yi_time_rate = lag_range[delay_win_range][rate_win_range_max_idx]
            yi_time_lag  = rate_range[rate_win_range][delay_win_range_max_idx]
        elif delay_win == False and rate_win == False :
            pass
        else :
            print("You should select the option of the both \"--delay-window\" and \"--rate-window\"!!")
            quit()
            
        fringe_lag_rate_00_complex_index1 = np.where(rate_range==yi_time_lag )[0][0] # the direction of the lag
        fringe_lag_rate_00_complex_index2 = np.where(lag_range ==yi_time_rate)[0][0] # the direction of the rate
        fringe_lag_rate_00_lag            = np.absolute(lag_rate_2D_array[fringe_lag_rate_00_complex_index1])
        fringe_lag_rate_00_rate           = np.absolute(lag_rate_2D_array[:,fringe_lag_rate_00_complex_index2])
        fringe_lag_rate_00_amp            = np.absolute(lag_rate_2D_array[fringe_lag_rate_00_complex_index1,fringe_lag_rate_00_complex_index2])
        fringe_lag_rate_00_phase          = np.angle(lag_rate_2D_array[fringe_lag_rate_00_complex_index1,fringe_lag_rate_00_complex_index2], deg=True)

        #
        # noise level, time domain
        #
        SNR_time_lag, noise_level_lag = noise_level(lag_rate_2D_array, fringe_lag_rate_00_amp)

    #
    # fringe output
    #
    if freq_plot == True : # cross-soectrum
        if l == 0 :
            ofile_name_freq = F"{save_file_name}_freq.txt"
            output_freq  = F"#******************************************************************************************************************************************************************************************\n"
            output_freq += F"#      Epoch        Label    Source      Length     Amp       SNR      Phase     Frequency     Noise-level           {station1_name}-azel               {station2_name}-azel                  MJD  \n"
            output_freq += F"#year/doy hh:mm:ss                        [s]       [%]                [deg]       [MHz]       1-sigma [%]   az[deg]  el[deg]  height[m]   az[deg]   el[deg]  height[m]          \n"
            output_freq += F"#******************************************************************************************************************************************************************************************"
            print(output_freq); output_freq += "\n"
        output1 = "%s    %s    %s     %.5f     %f %7.1f  %+8.3f    %8.3f      %f       %.3f  %.3f  %.3f       %.3f  %.3f  %.3f   %s" % \
            (epoch1, label, source_name, length*effective_integration_length, fringe_freq_rate_00_amp*100, SNR_freq_rate, fringe_freq_rate_00_phase2, fringe_freq_rate_00_freq, noise_level_freq*100, station1_azel[0], station1_azel[1], station1_azel[2], station2_azel[0], station2_azel[1], station2_azel[2], mjd)
        output_freq += "%s\n" % output1; print(output1)

    if freq_plot != True : # fringe
        if l == 0 :
            ofile_name_time = F"{save_file_name}_time.txt"
            output_time  = F"#****************************************************************************************************************************************************************************************************\n"
            output_time += F"#      Epoch         Label     Source      Length      Amp        SNR     Phase     Noise-level      Res-Delay     Res-Rate            {station1_name}-azel               {station2_name}-azel              MJD  \n"
            output_time += F"#year/doy hh:mm:ss                          [s]        [%]                [deg]     1-sigma[%]       [sample]        [Hz]      az[deg]  el[deg]  height[m]   az[deg]   el[deg]  height[m]          \n"
            output_time += F"#****************************************************************************************************************************************************************************************************"
            print(output_time); output_time += "\n"
        output2 = "%s    %s   %s     %.5f     %.6f  %7.1f  %+8.3f     %f        %+.2f      %+f       %.3f  %.3f  %.3f      %.3f  %.3f  %.3f   %s" % \
            (epoch1, label, source_name, length*effective_integration_length, fringe_lag_rate_00_amp*100, SNR_time_lag, fringe_lag_rate_00_phase, noise_level_lag*100, yi_time_rate, yi_time_lag, station1_azel[0], station1_azel[1], station1_azel[2], station2_azel[0], station2_azel[1], station2_azel[2], mjd)
        output_time += "%s\n" % output2; print(output2)

        if cumulate != 0 and add_plot != True :
            if l == 0 :
                cumulate_ofile_name = F"{cumulate_path}/{save_file_name}.png"
            cumulate_output += output_time
            cumulate_len.append(length)
            cumulate_snr.append(SNR_time_lag)
            cumulate_noise.append(noise_level_lag*100)
        
        if add_plot == True and cumulate == 0 :
            if l == 0 :
                add_plot_length = [i for i in range(length, length*loop+1, length)]
                add_plot_ofile_name = F"{add_plot_path}/{save_file_name}"
            #add_plot_length.append(length)
            add_plot_amp.append(fringe_lag_rate_00_amp*100)
            add_plot_snr.append(SNR_time_lag)
            add_plot_phase.append(fringe_lag_rate_00_phase)
            add_plot_noise_level.append(noise_level_lag*100)


    #
    # cross-spectrum
    #
    if freq_plot and time_plot :

        if rfi :
            fringe_time_freq_plot_path += F"rfi/freq_domain/len{length_label}s"
        else :
            fringe_time_freq_plot_path += F"/freq_domain/len{length_label}s"
        os.makedirs(fringe_time_freq_plot_path, exist_ok=True)


        #
        # make a graph
        #
        fig = plt.figure(figsize=(10, 7))
        gs  = GridSpec(nrows=3, ncols=2, height_ratios=[1,3,4])

        gs01 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0:2,0], hspace=0.0, height_ratios=[1,3])
        ax1 = fig.add_subplot(gs01[0,0])
        ax2 = fig.add_subplot(gs01[1,0])
        ax1.plot(freq_range, fringe_freq_rate_00_phase1  , lw=1)
        ax2.plot(freq_range, fringe_freq_rate_00_spectrum, lw=1)
        ax2.set_xlabel("Frequency [MHz]")
        ax1.set_ylabel("Phase")
        ax2.set_ylabel("Amplitude")
        ax1.set_xlim([0,BW])
        ax2.set_xlim([0,BW])
        ax1.set_ylim([-180,180])
        ax2.set_ylim(ymin=0)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticks([-90,0,90,180])
        ax1.grid(linestyle=":")
        ax2.grid(linestyle=":")

        gs3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[2,0])
        ax3 = fig.add_subplot(gs3[0,0])
        ax3.plot(rate_range, fringe_freq_rate_00_rate, lw=1)
        ax3.set_xlabel("Rate [Hz]")
        ax3.set_ylabel("Amplitude")
        ax3.set_xlim([rate_range[0],rate_range[-1]])
        ax3.set_ylim(ymin=0)
        ax3.grid(linestyle=":")

        gs4 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[0:2,1])
        ax4 = fig.add_subplot(gs4[0,0])
        #c = ax4.contourf(freq_range[j:j+chunk_size2], rate_range[i:i+chunk_size1], test[i:i+chunk_size1, j:j+chunk_size2], 100, cmap="rainbow")
        c = ax4.imshow(np.absolute(freq_rate_2D_array), extent=[freq_range[0], freq_range[-1], rate_range[-1], rate_range[0]], aspect="auto", cmap="rainbow")
        fig.colorbar(c)
        ax4.set_xlabel("Frequency [MHz]")
        ax4.set_ylabel("Rate [Hz]")
        ax4.set_xlim([0,BW])
        ax4.set_ylim([min(rate_range), max(rate_range)])
        ax4.grid(linestyle=":", color="black")

        gs5 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[2,1])
        ax5 = fig.add_subplot(gs5[0,0])
        data=[["Epoch", epoch1], ["Station-1", station1_name], ["Station-2", station2_name], \
              ["Source", source_name], ["Length [s]", "%.6f" % length], ["Frequency [MHz]", "%.6f" % observing_frequency], \
              ["Peak Amp [%]", "%.6f" % (fringe_freq_rate_00_amp * 100)], ["Peak Phs [deg]", "%.6f" % fringe_freq_rate_00_phase2] \
              , ["Peak Freq [MHz]", "%.6f" % fringe_freq_rate_00_freq], ["SNR", "%.6f" % SNR_freq_rate], ["1-sigma [%]", "%.8f" % (noise_level_freq*100)]]
        ax5.axis('tight')
        ax5.axis('off')
        table_ax5 = ax5.table(cellText=data, loc="center", cellLoc="left", colWidths=[0.73,0.78])
        table_ax5.set_fontsize(25)
        [table_ax5[i, 1].get_text().set_ha('right') for i in range(len(data))]
        for pos, cell in table_ax5.get_celld().items():
            cell.set_height(1/len(data))
            cell.set_linestyle('')

        plt.tight_layout()
        fig.savefig(F"{fringe_time_freq_plot_path}/{save_file_name}_freq_rate_search.png")
        #plt.show()
        plt.clf(); plt.close()

        if plane :
            dynamic_spectrum(freq_range, rate_range, np.absolute(freq_rate_2D_array), "Frequency [MHz]", "Rate [Hz]", "cross-spectrum [%]", \
                             [min(freq_range), max(freq_range)], [min(rate_range), max(rate_range)], F"{plane_path}/{save_file_name}_freq_rate_plane_expansion.png")
        if fits :
            from astropy.io import fits
            hdu = fits.PrimaryHDU(np.absolute(freq_rate_2D_array))
            hdr = hdu.header
            hdr['CRPIX1'] = len(freq_range)               # x 軸の基準ピクセル
            hdr['CRVAL1'] = freq_range[-1]                # x 軸の基準値
            hdr['CDELT1'] = freq_range[1] - freq_range[0] # x 軸のピクセルスケール（例：度、mm など）
            hdr['CTYPE1'] = 'Frequency [MHz]'             # x 軸のタイプ
            hdr['CRPIX2'] = len(rate_range)               # y 軸の基準ピクセル
            hdr['CRVAL2'] = rate_range[-1]                # y 軸の基準値
            hdr['CDELT2'] = rate_range[1] - rate_range[0] # y 軸のピクセルスケール
            hdr['CTYPE2'] = 'Rate [Hz]'                   # y 軸のタイプ
            hdu.writeto(f"{fits_path}/{save_file_name}_freq_rate_plane.fits", overwrite=True)


    #
    # delay-rate search window
    #
    if freq_plot != True and time_plot :
        
        # When a RFI is quite large in the fringe search area in the time domain 
        #if cmap_time == True :
        cmap_blow_up_delay = (delay_win_range_low <= lag_range)  & (lag_range <= delay_win_range_high)
        cmap_blow_up_rate  = (rate_win_range_low <= rate_range) & (rate_range <= rate_win_range_high)
        
        
        #
        # a directory to save a graph
        #
        if rfi :
            fringe_time_freq_plot_path += F"rfi/time_domain/len{length_label}s"
        else :
            fringe_time_freq_plot_path += F"/time_domain/len{length_label}s"
        os.makedirs(fringe_time_freq_plot_path, exist_ok=True)

        fig  = plt.figure(figsize=(10, 7))
        grid = plt.GridSpec(2, 2)

        ax1 = fig.add_subplot(grid[0,0]) # delay
        ax2 = fig.add_subplot(grid[1,0]) # rate
        ax3 = fig.add_subplot(grid[0,1]) # delay-rate search window

        ax1.plot(lag_range, fringe_lag_rate_00_lag, lw=1)
        ax1.set_xlabel("Delay [Sample]")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim([delay_win_range_low, delay_win_range_high])
        ax1.set_ylim(ymin=0)
        ax1.yaxis.set_major_formatter('{x:.1e}')
        ax1.grid(linestyle=":")

        ax2.plot(rate_range, fringe_lag_rate_00_rate, lw=1)
        ax2.set_xlabel("Rate [Hz]")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim([min(rate_range), max(rate_range)])
        ax2.set_ylim(ymin=0)
        ax2.yaxis.set_major_formatter('{x:.1e}')
        ax2.grid(linestyle=":")

        if cmap_time :
            #c = ax3.contourf(lag_range[cmap_blow_up_delay], rate_range[cmap_blow_up_rate], np.absolute(lag_rate_2D_array[cmap_blow_up_rate][:,cmap_blow_up_delay]), 100, cmap="rainbow", vmin=0.0)
            c = ax3.imshow(np.absolute(lag_rate_2D_array[cmap_blow_up_rate][:,cmap_blow_up_delay]), extent=[lag_range[cmap_blow_up_delay][0], lag_range[cmap_blow_up_delay][-1], rate_range[cmap_blow_up_rate][-1], rate_range[cmap_blow_up_rate][0]], interpolation="gaussian", aspect="auto", cmap="rainbow", vmin=0.0)
        else :
            #c = ax3.contourf(lag_range[cmap_blow_up_delay], rate_range[cmap_blow_up_rate], np.absolute(lag_rate_2D_array[cmap_blow_up_rate][:,cmap_blow_up_delay]), 100, cmap="rainbow", vmin=0.0, vmax=np.amax(np.absolute(lag_rate_2D_array)))
            c = ax3.imshow(np.absolute(lag_rate_2D_array[cmap_blow_up_rate][:,cmap_blow_up_delay]), extent=[lag_range[cmap_blow_up_delay][0], lag_range[cmap_blow_up_delay][-1], rate_range[cmap_blow_up_rate][-1], rate_range[cmap_blow_up_rate][0]], interpolation="gaussian", aspect="auto", cmap="rainbow", vmin=0.0, vmax=np.amax(np.absolute(lag_rate_2D_array)))
            #c = ax3.contourf(lag_range, rate_range, np.absolute(lag_rate_2D_array), 100, cmap="rainbow", vmin=0.0)

        if delay_win == False and rate_win == False :
                delay_win_range_low, delay_win_range_high = -10, 10
                
        fig.colorbar(c, format="%.1e")
        #if delay_win != False and rate_win != False :
        #    ax3.plot(lag_range[fringe_lag_rate_00_complex_index2], rate_range[fringe_lag_rate_00_complex_index1], "x", color="white")
        ax3.set_xlabel("Delay [Sample]")
        ax3.set_ylabel("Rate [Hz]")
        ax3.set_xlim(delay_win_range_low, delay_win_range_high)
        ax3.set_ylim(rate_win_range_low, rate_win_range_high)
        ax3.grid(linestyle=":", color="black")

        ax4 = fig.add_subplot(grid[1,1])
        ax4.axis('tight')
        ax4.axis('off')
        data=[["Epoch", epoch1], ["Station-1", station1_name], ["Station-2", station2_name], \
              ["Source", source_name], ["Length [s]", "%.6f" % length], ["Frequency [MHz]", "%.6f" % observing_frequency], \
              ["Peak Amp [%]", "%.6f" % (fringe_lag_rate_00_amp * 100)], ["Peak Phs [deg]", "%.6f" % fringe_lag_rate_00_phase], \
              ["SNR", "%.6f" % SNR_time_lag], ["1-sigma [%]", "%.8f" % (noise_level_lag*100)], ["delay [sample]", "%+.1e" % yi_time_rate], ["rate [mHz]", "%+.3f" % (yi_time_lag*1000)]]
        table_ax4 = ax4.table(cellText=data, loc="center", cellLoc="left", colWidths=[0.65,0.65])
        table_ax4.set_fontsize(30)
        [table_ax4[i, 1].get_text().set_ha('right') for i in range(len(data))]
        for pos, cell in table_ax4.get_celld().items():
            cell.set_height(1.0/len(data))
            cell.set_linestyle('')

        plt.tight_layout()
        fig.savefig(F"{fringe_time_freq_plot_path}/{save_file_name}_delay_rate_search.png")
        #plt.show()
        plt.clf(); plt.close()

        if plane :
            dynamic_spectrum(lag_range , rate_range , np.absolute(lag_rate_2D_array), "Delay [Sample]", "Rate [Hz]", "cross-spectrum [%]", \
                             [min(lag_range), max(lag_range)], [min(rate_range), max(rate_range)], F"{plane_path}/{save_file_name}_delay_rate_plane_expansion.png")

            if fits :
                from astropy.io import fits
                hdu = fits.PrimaryHDU(np.absolute(lag_rate_2D_array))
                hdr = hdu.header
                hdr['CRPIX1'] = len(lag_range)                # x 軸の基準ピクセル
                hdr['CRVAL1'] = lag_range[-1]                 # x 軸の基準値
                hdr['CDELT1'] = lag_range[1] - lag_range[0]   # x 軸のピクセルスケール（例：度、mm など）
                hdr['CTYPE1'] = "Delay [Sample]"              # x 軸のタイプ
                hdr['CRPIX2'] = len(rate_range)               # y 軸の基準ピクセル
                hdr['CRVAL2'] = rate_range[-1]                # y 軸の基準値
                hdr['CDELT2'] = rate_range[1] - rate_range[0] # y 軸のピクセルスケール
                hdr['CTYPE2'] = "Rate [Hz]"                   # y 軸のタイプ
                hdu.writeto(f"{fits_path}/{save_file_name}_delay_rate_plane.fits", overwrite=True)


    #
    # frequency-rate 2D array & lag-rate 2D array output
    #
    #"""        
    if cs_output :
        import pandas as pd 
        print(freq_rate_2D_array.T)
        freq_rate_2D_array_df = pd.DataFrame(freq_rate_2D_array.T, dtype=complex)
        lag_rate_2D_array_df  = pd.DataFrame(lag_rate_2D_array, dtype=complex)

        freq_rate_2D_array_df_file_name = "%s/%s_freq_rate_search.csv" % (save_file_path, save_file_name)
        lag_rate_2D_array_df_file_name  = "%s/%s_delay_rate_search.csv" % (save_file_path, save_file_name)

        freq_rate_2D_array_df.index   = ["%.0f" % i for i in freq_range]
        freq_rate_2D_array_df.columns = ["0" if i == 0.0 else "%.5f" % i for i in rate_range]
        freq_rate_2D_array_df.to_csv(freq_rate_2D_array_df_file_name)

        lag_rate_2D_array_df.index   = [i for i in rate_range]
        lag_rate_2D_array_df.columns = ["%.0f" % i for i in lag_range]
        lag_rate_2D_array_df.to_csv(lag_rate_2D_array_df_file_name)
    #"""

    #
    # 2D-graph
    # Delay-rate Search
    #
    if DDD == True :        
        
        from scipy import interpolate

        delay_rate_search_3D_lag  = np.array([lag_range]  * lag_rate_2D_array.shape[0])
        delay_rate_search_3D_rate = np.array([rate_range] * lag_rate_2D_array.shape[1]).T

        for i in range(2) :

            fig = plt.figure(figsize = (8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("Delay [sample]", size = 14)
            ax.set_ylabel("Rate [Hz]"     , size = 14)
            ax.set_zlabel("Amplitude [%]" , size = 14)
            ax.view_init(elev=10, azim=45)
            plt.tight_layout()
            
            if i == 0 :
                ax.plot_surface(delay_rate_search_3D_lag, delay_rate_search_3D_rate, np.absolute(lag_rate_2D_array), cmap="jet")
                plt.savefig(F"{ddd_path}/{save_file_name}_delay_rate_search_3D_GlobalImage.pdf")
                plt.close()
            elif i == 1 : # In the enlargement graph, adjust the xrange and yrange.
                delay_rate_search_3D_lag[-15 >= delay_rate_search_3D_lag]    = np.nan
                delay_rate_search_3D_lag[+15 <= delay_rate_search_3D_lag]    = np.nan
                delay_rate_search_3D_rate[-0.1 >= delay_rate_search_3D_rate] = np.nan
                delay_rate_search_3D_rate[+0.1 <= delay_rate_search_3D_rate] = np.nan

                ax.plot_surface(delay_rate_search_3D_lag, delay_rate_search_3D_rate, np.absolute(lag_rate_2D_array), cmap="jet")
                plt.savefig(F"{ddd_path}/{save_file_name}_delay_rate_search_3D_Enlargement.pdf")
                plt.close()
#
# output
#
if output == True :
    if rfi == True :
        fringe_time_freq_output_path += F"rfi/time_domain/len{length_label}s"
    else :
        fringe_time_freq_output_path += F"/time_domain/len{length_label}s"
        if freq_plot == True :
            fringe_time_freq_output_path = fringe_time_freq_output_path.replace("/time_domain/", "/freq_domain/")
    os.makedirs(fringe_time_freq_output_path, exist_ok=True)

    if freq_plot != True : # time domain
        ofile_time = open(F"{fringe_time_freq_output_path}/{ofile_name_time}", "w")
        ofile_time.write(output_time)
        ofile_time.close()
    if freq_plot == True : # freq domain
        ofile_freq = open(F"{fringe_time_freq_output_path}/{ofile_name_freq}", "w")
        ofile_freq.write(output_freq)
        ofile_freq.close()
else :
    pass

#
# the cumulation of the integration time
#
if cumulate != 0 and add_plot != True :

    from scipy.optimize import curve_fit

    cumulate_path += "/%s_%s" % (source_name, save_file_name.split("_")[2])
    os.makedirs(cumulate_path, exist_ok=True)

    ofile_cumu = open(F"{cumulate_path}/{save_file_name}.txt", "w")
    ofile_cumu.write(cumulate_output)
    ofile_cumu.close()

    def power_law_equation(x, a, b) :
        y = a * (x)**b
        return y
    
    # length vs SNR
    param, conv = curve_fit(power_law_equation, cumulate_len, cumulate_snr)
    x_data  = np.linspace(cumulate_len[0], cumulate_len[-1], int((cumulate_len[-1] - cumulate_len[0])/10))
    y_data1 = power_law_equation(x_data, *param)
    y_data2 = power_law_equation(x_data,  param[0], 0.5)


    fig_cumu, axs_cumu = plt.subplots(2, 1, figsize=(12,8), tight_layout=True, height_ratios=[1, 1])
    axs_cumu[0].loglog(cumulate_len, cumulate_snr,"o", color="tab:green", label="Source: %s\nInterval: %d s" %(source_name, cumulate))
    axs_cumu[0].loglog(x_data, y_data1, "-",  color="tab:red", label="Power-law fit\nIndex (obseved): %.3f" % param[1])
    axs_cumu[0].loglog(x_data, y_data2, "--", color="tab:red", label="Index (theoretical): 0.5")
    axs_cumu[0].set_xlabel("logarithm integration time [s]")
    axs_cumu[0].set_ylabel("S/N")
    axs_cumu[0].set_xlim(cumulate_len[0],cumulate_len[-1])
    axs_cumu[0].set_ylim(ymin=int(min(cumulate_snr)))
    axs_cumu[0].legend(loc="upper left")
    axs_cumu[1].plot(cumulate_len, cumulate_snr, "o", color="tab:green")
    axs_cumu[1].plot(x_data, y_data1, "-" , color="tab:red")
    axs_cumu[1].plot(x_data, y_data2, "--", color="tab:red")
    axs_cumu[1].set_xlabel("Integration time [s]")
    axs_cumu[1].set_ylabel("S/N")
    axs_cumu[1].set_xlim(cumulate_len[0],cumulate_len[-1])
    axs_cumu[1].set_ylim(ymin=int(min(cumulate_snr)))
    #axs_cumu[1].legend(loc="upper left")
    plt.savefig(F"{cumulate_path}/{save_file_name}.png")
    plt.clf(); plt.close()


    # Noise-level
    param, conv = curve_fit(power_law_equation, cumulate_len, cumulate_noise)
    x_data = np.linspace(cumulate_len[0], cumulate_len[-1], int((cumulate_len[-1] - cumulate_len[0])/10))
    y_data = power_law_equation(x_data, *param)
    
    fig = plt.figure(dpi=100, figsize=(10.24,5.12))
    plt.loglog(cumulate_len, cumulate_noise, "o", label="Source: %s\nInterval: %d s" %(source_name, cumulate))
    plt.loglog(x_data, y_data, label="Power-law fit\nIndex: %.3f" % param[1])
    plt.xlabel("Integration time [s]")
    plt.ylabel("Noise-level [%]")
    plt.xlim(cumulate_len[0],cumulate_len[-1])
    #plt.ylim(ymin=int(min(cumulate_noise)))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(F"{cumulate_path}/{save_file_name}_noise-level.png")
    #plt.show()
    plt.clf(); plt.close()


#
# plot the amplitude, the phase, the S/N and the noise level in the time domain
#
if add_plot == True and cumulate == 0 :
    save_file_name = add_plot_ofile_name.split("/")[-1].replace("_time.txt", "")
    epoch1 = ofile_name_time.split("_")[2]
    add_plot_path += "/%s_%s" % (source_name, epoch1)
    os.makedirs(add_plot_path, exist_ok=True)

    time_add_plot(source_name, add_plot_length, add_plot_amp        , epoch1, "Amplitude [%]" , "amp"   , add_plot_length[0], add_plot_length[-1], "%s/%s" % (add_plot_path, save_file_name))
    time_add_plot(source_name, add_plot_length, add_plot_snr        , epoch1, "S/N"            , "snr"  , add_plot_length[0], add_plot_length[-1], "%s/%s" % (add_plot_path, save_file_name))
    time_add_plot(source_name, add_plot_length, add_plot_phase      , epoch1, "Phase [deg]"    , "phase", add_plot_length[0], add_plot_length[-1], "%s/%s" % (add_plot_path, save_file_name))
    time_add_plot(source_name, add_plot_length, add_plot_noise_level, epoch1, "Noise Level [%]", "noise", add_plot_length[0], add_plot_length[-1], "%s/%s" % (add_plot_path, save_file_name))


#
# dynamic spectrum: the frequency domain & the time domain
#
if ds_plot == True :
    dynamic_spectrum_freq_time = []
    dynamic_spectrum_lag_time = []

    for i in range(fft_point//2) :
        dynamic_spectrum_freq_time.append(complex_visibility[:,i]) # lag vs integ time in the dynamic spectrum

    for i in range(PP) :
        ifft_time_direction = np.fft.ifft(complex_visibility[i], n=fft_point) # convert the frequency domain to the time domain by executing the IFFT in the frequency
        dynamic_spectrum_lag_time.append(ifft_time_direction)

    dynamic_spectrum_freq_time = np.array(dynamic_spectrum_freq_time)
    dynamic_spectrum_lag_time  = np.array(dynamic_spectrum_lag_time)
    dynamic_spectrum_lag_time  = np.concatenate([dynamic_spectrum_lag_time[:, fft_point//2:], dynamic_spectrum_lag_time[:, :fft_point//2]], 1)
    dynamic_spectrum_lag_time  = dynamic_spectrum_lag_time[:, ::-1]
    
    time = save_file_name.split("_")[2]
    time = datetime.datetime.strptime(time, "%Y%j%H%M%S")
    time = time.strftime("%Y/%j %H:%M:%S")

    dynamic_spectrum(freq_range, integ_range, dynamic_spectrum_freq_time.T, "frequency [MHz]"  , "elapesd time (sec) since %s UT" % time, "amplitude", [min(freq_range), max(freq_range)], [0,PP], "%s/%s_dynamic_spectrum_frequency.png" %  (dynamic_spectrum_path, save_file_name)) # the frequency domain: length vs frequency and cross-spectrum
    dynamic_spectrum(lag_range , integ_range, dynamic_spectrum_lag_time   , "time lag [samles]", "elapesd time (sec) since %s UT" % time, "amplitude", [min(lag_range) , max(lag_range)] , [0,PP], "%s/%s_dynamic_spectrum_time_lag.png"  %  (dynamic_spectrum_path, save_file_name)) # the time domain: length vs the frequency domain


