import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Input data: List of comma-separated feature strings
payload = [
    {"input": "0.209377,3109.03329,85.200147,22.394407,8.138688,0.699861,0.025578,9.812214,5.555634,4126.58731,22.5984,175.638726,152.707705,823.928241,257.432377,47.223358,0.56348064,23.3876,4.8519155,0.023482,1.050225,0.069225,13.784111,1.302012,36.205956,69.0834,295.570575,0.23868,0.284232,89.24556,84.31664,29.657104,5.31069,1.74307,23.187704,7.294176,1.987283,1433.16675,0.949104,30.87942,78.526968,3.828384,13.39464,10.265073,9028.291921,3.58345,7.298161939,1.73855,0.094822,11.339138,72.611063,2003.810319,22.136229,69.834944,0.120342857"},
    {"input": "0.145282,978.76416,85.200147,36.968889,8.138688,3.63219,0.025578,13.51779,1.2299,5496.92824,19.4205,155.86803,14.75472,51.216883,257.432377,30.284345,0.48471012,50.628208,6.085041,0.031442,1.113875,1.1178,28.310953,1.357182,37.476568,70.79836,178.5531,0.23868,0.363489,110.581815,75.74548,37.532,0.0055176,1.74307,17.222328,4.926396,0.858603,1111.28715,0.003042,109.125159,95.415086,52.26048,17.175984,0.29685,6785.003474,10.358927,0.173229,0.49706,0.568932,9.292698,72.611063,27981.56275,29.13543,32.131996,21.978"},
    {"input": "0.47003,2635.10654,85.200147,32.360553,8.138688,6.73284,0.025578,12.82457,1.2299,5135.78024,26.4825,128.988531,219.32016,482.141594,257.432377,32.563713,0.49585158,85.955376,5.376488,0.036218,1.050225,0.70035,39.364743,1.009611,21.459644,70.8197,321.426625,0.23868,0.210441,120.0564375,65.46984,28.053464,1.289739,1.74307,36.861352,7.813674,8.146651,1494.076488,0.377208,109.125159,78.526968,5.390628,224.207424,8.745201,8338.906181,11.626917,7.70956011,0.97556,1.198821,37.077772,88.609437,13676.95781,28.022851,35.192676,0.196941176"},
    {"input": "0.252107,3819.65177,120.201618,77.112203,8.138688,3.685344,0.025578,11.053708,1.2299,4169.67738,23.6577,237.282264,11.05041,661.51864,257.432377,15.201914,0.71788212,88.15936,2.347652,0.029054,1.4003,0.636075,41.1169605,0.722727,21.530392,47.27586,196.607985,0.23868,0.292431,139.82457,71.5712,24.354856,2.655345,1.74307,52.003884,7.38606,3.813326,15691.55218,0.614484,31.674357,78.526968,31.323372,59.301984,7.884336,10965.76604,14.852022,6.122161814,0.49706,0.284466,18.529584,82.416803,2094.262452,39.948656,90.493248,0.155828571"},
    {"input": "0.380297,3733.04844,85.200147,14.103738,8.138688,3.942255,0.05481,3.396778,102.15198,5728.73412,24.0108,324.546318,149.717165,6074.859475,257.432377,82.213495,0.53646696,72.644264,30.537722,0.025472,1.050225,0.69315,31.724726,0.82755,34.41536,74.06532,200.17816,0.23868,0.207708,97.92012,52.83888,26.019912,1.144902,1.74307,9.064856,7.35072,3.490846,1403.6563,0.164268,109.125159,91.994825,51.141336,29.10264,4.27464,16198.04959,13.666727,8.153057673,48.50134,0.121914,16.408728,146.109943,8524.370502,45.381316,36.262628,0.096614458"},
    {"input": "0.209377,2615.8143,85.200147,8.541526,8.138688,4.013127,0.025578,12.547282,1.2299,5237.54088,10.2399,148.487931,16.52612,642.325163,257.432377,18.382,0.63946044,80.6674,14.68803,0.016716,1.050225,0.857625,32.456996,1.390284,7.03064,55.22404,135.48925,0.23868,0.478275,135.317865,81.46312,31.7316,0.0055176,1.74307,16.773128,4.926396,2.394414,866.38295,0.003042,109.125159,78.526968,3.828384,23.30496,0.29685,8517.278846,10.981896,0.173229,0.49706,1.164956,21.915512,72.611063,24177.59555,28.525186,82.527764,21.978"},
    {"input": "0.3482495,1733.65412,85.200147,8.377385,15.31248,1.913544,0.025578,6.547778,1.2299,5710.46099,17.655,143.646993,344.644105,719.7251425,257.432377,38.455144,0.94632336,78.304856,13.1842555,0.033631,1.050225,0.61095,13.784111,2.786085,21.877508,19.2157,107.907985,1.318005,0.4605105,176.6255625,97.07586,44.506128,1.006962,1.74307,4.474032,4.926396,2.62015,1793.612375,0.097344,13.214487,78.526968,26.304948,48.2664,1.460502,3903.806766,10.777915,4.408484219,0.8613,0.467337,17.878444,192.453107,3332.467494,34.166222,100.086808,0.06509589"},
    {"input": "0.269199,966.45483,85.200147,21.174189,8.138688,4.987617,0.025578,9.408886,1.2299,5040.77914,20.8329,170.051724,6.1999,701.01861,257.432377,12.8674,0.77190948,71.542272,24.9103515,0.02985,1.050225,1.109625,41.9329185,1.186155,42.11346,63.21684,326.225295,0.23868,0.325227,83.7693675,73.988,19.905608,2.117379,1.74307,15.870236,8.354376,3.277203,767.7205625,0.292032,15.089217,104.9853285,5.104512,37.5564,4.518057,18090.34945,10.342388,6.591895977,0.49706,0.277693,18.445866,109.6939865,21371.75985,35.208102,31.424696,0.092872964"}
]

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Send POST request
for request in payload:
    response = requests.post(url, json=request, headers=headers)
    print("Status code:", response.status_code)
    print("Predictions:", response.json())
    print("\n")
