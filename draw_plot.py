import matplotlib.pyplot as plt

ga_avg_score = [-4487.5, -4153.115092852569, -3722.434749519728, -3340.7989660182366, -3029.612533836556, -2727.01978497696, -2345.554222284908, -1988.6813039012043, -1665.8437494956875, -1431.2575264534248, -1251.2110347727687, -1089.420225830523, -986.9987279599239, -899.1953760729104, -828.4471477733208, -793.4071042078006, -768.4835317080565, -732.8315524877237, -703.6696469810676, -675.6261661366868, -644.876019615401, -624.4428918066471, -595.7762887823335, -577.3243163141508, -562.6569599898336, -531.9034885202891, -487.4339869009515, -456.0527521228181, -437.1324813838616, -417.7488563451644, -399.83937321931825, -373.9780728153241, -350.36161465830776, -332.03388287506806, -310.13571868706777, -293.00412128163686, -272.7870719751811, -247.4565580223138, -222.05326473247288, -189.57010393625464, -158.27451926035343, -135.0307824443487, -123.6401423932703, -106.51039811123772, -76.31654583186403, -53.79239572781275, -28.005041131800407, -4.392622341449962, 25.83541599675342, 46.74318522986046, 58.96560451105108, 73.92038155021729, 95.675087423181, 112.26136372909878, 137.62692310969214, 165.93150577706513, 189.8106794026893, 216.79883542599828, 245.23945056122676, 261.3348747470873, 283.7842330684447, 301.1467891637982, 317.32248567188515, 336.7407149175455, 357.37634750496153, 380.1190392659845, 405.6726094364294, 425.12258976517035, 442.0193878141837, 458.76347340834917, 478.13818336094016, 494.78158903309793, 507.7916055410092, 520.253532505659, 535.8178577112334, 545.0850851715248, 561.1032103925115, 576.8928057354238, 596.1443009655643, 611.3178575211228, 634.8224689937118, 654.1421371337003, 665.4278845205174, 681.9146019008527, 690.8176716745163, 700.8296137364475, 708.6378919472018, 722.05838024352, 731.7515377895631, 744.940906298658, 755.9568337415001, 769.5678207314603, 781.1015191720538, 792.3220529524016, 801.9420554947283, 816.3521514054241, 827.2387498415428, 837.7675826569775, 845.4836564933778, 857.1045184212146]
ga_best_score = [-4487.5, -3863.2524752475247, -3405.0544554455446, -2900.856435643564, -2377.3690503601856, -2187.9795822865867, -1925.1457317373533, -1540.343258952761, -1309.3020183284755, -1145.78958051753, -1068.5826136414403, -925.1535034769901, -864.3991442059122, -790.1663708583676, -784.4292941835968, -757.137867772969, -733.7072559954851, -692.5229855528829, -621.7882989373435, -621.7882989373435, -608.4880176667638, -587.9069265005375, -535.5414637028182, -535.5414637028182, -524.6523857606917, -456.44379901464305, -432.2214593122523, -413.7095023874253, -395.9491296076103, -369.25847746911484, -355.3892958563483, -346.64371086686145, -309.10269983738306, -300.2848102842359, -281.5574091208882, -240.95083076125474, -237.44062221518703, -206.54926818712423, -171.38637448775322, -154.42726802239082, -127.08491940586104, -105.55993503260687, -93.101810262722, -43.011273328313045, -35.85660199441695, -13.255842452401758, 25.417912723281702, 43.46194658960008, 57.329199445216204, 81.76122433337108, 85.55904393801282, 128.52390576380168, 128.52390576380168, 161.4635628478545, 186.19193786034253, 236.4113610840514, 261.48555290428305, 287.37479048238447, 287.37479048238447, 304.24311571367537, 311.84748847489885, 330.14099024228767, 346.30487644888825, 388.97618758606166, 417.78509035082476, 449.99216512712434, 449.99216512712434, 449.99216512712434, 476.95589246077645, 489.8173920234683, 518.1757924050528, 520.6535775553908, 526.0513429033084, 542.9783926271631, 564.3596632845614, 567.7935924627441, 608.9893129305325, 617.9899243462955, 633.1729908869257, 649.4230433035284, 667.7993505063534, 690.8258634609161, 700.1533894719382, 706.2937088857421, 712.1049119532315, 714.9311871701121, 747.6640424049723, 754.5757325724461, 754.5757325724461, 777.59440718472, 788.7853660928935, 799.7832636559996, 799.7832636559996, 819.0312780249383, 831.989447202975, 863.9875680750465, 863.9875680750465, 866.1744408011259, 866.1744408011259, 890.3496051436538]

sa_avg_score = [-2172.1773731719804, -1213.4172800206686, -694.6962990706663, -355.57236626882906, -122.95103435328986, 61.239599503885266, 205.08264069062074, 322.2942366237697, 416.24346391500814, 998.8190215765273, 1128.4909356110452, 1203.072746903143, 1263.8042282129275, 1310.800031411509, 1357.1649612415881, 1395.00750405426, 1429.9086933193003, 1461.0239037502135, 1491.9591636364078, 1515.885188078727, 1536.684426848789, 1557.5153075564835, 1576.4601444967534, 1596.6159523271451, 1614.5457118241952, 1628.5480717761031, 1641.9882778037816, 1654.4141551564685, 1668.498419609822, 1683.0447399314348, 1690.5569641867028, 1697.0318839841661, 1703.706803781629, 1708.9607586842553, 1713.412247817237, 1717.6907457705154, 1721.1018504736032, 1724.5129551766909, 1726.465672779112, 1728.8132726641772, 1731.333872121028, 1733.8544715778785, 1736.175071034729, 1738.4956704915792, 1741.130865671108, 1743.766060850637, 1747.2700682212494, 1750.9408229235828, 1754.4115776259168, 1756.3877229133182, 1758.190868628934, 1761.2568866989702, 1764.6816610548012, 1768.1064354106322, 1772.8312386206387, 1777.5560418306457, 1781.4174590766524, 1785.1121289909377, 1788.8067989052233, 1792.7069002995536, 1796.607001693884, 1799.2442307337938, 1801.5227034879088, 1804.85882368298, 1806.3164807440976, 1807.7741378052149, 1809.0372167262162, 1810.3002956472178, 1811.563374568219, 1812.6210220091755, 1814.4990118944054, 1816.5953833329484, 1820.1254097734502, 1822.5977887729955, 1825.270167772541, 1827.942546772087, 1830.6149257716327, 1834.5110814492054, 1838.6072371267783, 1842.9197140002639, 1846.611848429476, 1850.085601305374, 1852.1256991793143, 1854.1657970532542, 1856.4345856620057, 1858.7033742707576, 1860.9721628795094, 1862.017174810234, 1862.862186740959, 1863.4908774757707, 1863.9195682105826, 1864.3482589453947, 1864.7769496802066, 1865.2056404150185, 1865.2056404150185, 1865.2056404150185, 1865.4074025964353, 1865.8339917709673, 1866.2605809454994, 1866.6871701200314, 1867.3607648383033, 1868.0343595565748, 1868.7079542748463, 1869.3815489931178, 1870.0551437113893, 1870.728738429661, 1871.200570966516, 1871.4475765102554, 1872.2920699300787, 1873.1365633499022, 1874.994110710672, 1876.8516580714415, 1879.318255695746, 1881.7848533200504, 1884.2514509443547, 1886.718048568659, 1889.184646192963, 1891.6512438172674, 1893.5203535654878, 1895.3894633137074, 1896.589657483605, 1897.7898516535024, 1898.92657729569, 1900.063302937878, 1901.2000285800655, 1902.3367542222536, 1903.4734798644415, 1904.610205506629]
sa_best_score = [143.14525365603885, 704.1029062819555, 861.4666437793403, 1000.9233649385201, 1040.155625224406, 1166.383402646936, 1211.983928997769, 1259.9870040889614, 1261.786509536154, 1338.2555766151931, 1439.864394001216, 1449.9210192029368, 1468.7814568771828, 1470.8813969243363, 1503.8049235251956, 1544.808830773652, 1560.9958216481748, 1571.1391083980945, 1571.1391083980945, 1577.5158210383859, 1647.856781701836, 1658.2298262798836, 1658.2298262798836, 1672.4394752282526, 1683.1025184956968, 1684.832430292728, 1695.397881924961, 1695.397881924961, 1711.981752931631, 1722.979024254515, 1722.979024254515, 1722.979024254515, 1724.979024254515, 1724.979024254515, 1727.617409825513, 1727.617409825513, 1729.5089289558396, 1729.5089289558396, 1731.5089289558396, 1746.455023105167, 1748.1850188230203, 1748.1850188230203, 1748.1850188230203, 1748.1850188230203, 1753.9693616208012, 1753.9693616208012, 1764.5490026619636, 1766.2164759791776, 1766.2164759791776, 1766.2164759791776, 1766.2164759791776, 1778.8451995233847, 1782.4327623813306, 1782.4327623813306, 1801.2173937208695, 1801.2173937208695, 1803.163175122031, 1803.163175122031, 1803.163175122031, 1805.2174899224815, 1805.2174899224815, 1805.2174899224815, 1805.2174899224815, 1815.793964332044, 1815.793964332044, 1815.793964332044, 1815.793964332044, 1815.793964332044, 1815.793964332044, 1815.793964332044, 1823.9973887747822, 1826.1812043079124, 1840.5177543275, 1840.5177543275, 1842.5177543275, 1842.5177543275, 1842.5177543275, 1854.7555211077715, 1856.7555211077715, 1858.9187330668994, 1860.9187330668994, 1860.9187330668994, 1860.9187330668994, 1860.9187330668994, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1865.2056404150187, 1867.223262229184, 1869.4715321603398, 1869.4715321603398, 1869.4715321603398, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1871.941587597734, 1877.916466358575, 1877.916466358575, 1890.5170612054308, 1890.5170612054308, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1896.607563840777, 1902.519002904405, 1902.519002904405, 1907.9748202626545, 1907.9748202626545, 1907.9748202626545, 1907.9748202626545, 1907.9748202626545, 1907.9748202626545]

ts_avg_score = [-3987.30198019802, -3487.10396039604, -3105.108245899217, -2775.886040096547, -2514.3398403310357, -2294.221267655529, -2122.1818344259427, -1982.4688200336868, -1863.0463123512727, -1760.8769797243092, -1672.023496634418, -1593.035162292449, -1520.3229247631646, -1455.9979444622127, -1397.0273919242502, -1343.848244266453, -1294.0901728107203, -1247.3250837735957, -1203.9831420370585, -1162.7813286949636, -1124.0243687610835, -1087.8589261723907, -1054.0287246585372, -1020.9155179979078, -988.7551687329012, -958.3618457696568, -928.3732639256978, -899.0477042933434, -870.3659177752319, -842.120252097682, -814.5948680517046, -787.9856386752112, -761.9236832545719, -736.909072109823, -712.1547582093125, -687.806657567621, -664.4802475919001, -641.6218225412497, -618.8905347179781, -596.2728976568566, -574.2041919395253, -552.2979326526468, -530.7465910452122, -509.39221200061365, -488.76087144618316, -468.75309992805813, -448.98934361279885, -429.6615291566952, -410.41653302231106, -391.62567068535554, -373.26334419408784, -354.94480392920553, -337.1575266945139, -319.6872675131335, -302.62352040383166, -285.5554535372392, -269.0420038346531, -252.93402675039096, -237.20985440288405, -221.8493086614981, -206.43205210531008, -191.42042817989878, -176.9853437642528, -162.51717261601644, -148.12217245740132, -134.01907255498327, -119.91615121321905, -106.10466891499578, -92.37251407477531, -79.03875550861325, -65.86877923557505, -52.811850200337275, -39.98393287183149, -27.51010984740681, -15.32646319966208, -3.145417870400219, 9.0841413509674, 21.04369244663645, 33.11752040679952, 44.931421621411815, 56.56880403786812, 68.06991815544502, 79.77794921383452, 91.42338911482949, 102.83397680336775, 114.12049847397235, 125.28691950521967, 136.33780630510216, 147.3890990508903, 158.3966085768782, 169.13344875033917, 179.66086412397618, 190.4315124971334, 201.06015994132812, 211.48763239494454, 221.72065567047426, 231.9296104545499, 241.96349742507886, 251.53772670545578, 308.7576476164501, 355.9526899344419, 393.25635250384374, 425.46474120737594, 452.7318088226646, 477.4052994750355, 499.9378604389123, 521.8452047052543, 543.3104398738374, 564.0307499786999, 584.1306765640421, 604.1848713609818, 623.7752654035111, 642.6847251249028, 661.4181282644062, 679.8421183687003, 698.3411699981916, 716.7833467086575, 734.7585670121668, 752.3648834444846, 769.7819569394056, 787.0932291187141, 804.2456577649424, 821.2657454419876, 837.8489545896932, 854.1379625268432, 870.1914183467442, 885.8507177557495, 901.1938968435492, 916.0635777414836, 930.473988234, 944.4209510659749, 958.5302769848732, 972.2243378671407, 985.8326756671516, 998.9050524456834, 1011.6821709918852, 1024.3713083074688, 1036.6357844976462, 1048.5939369941052, 1059.9634221462047, 1071.4819893469469, 1082.7492451084474, 1093.9440639296054, 1104.6267269874024, 1115.815174339222, 1126.7247316313271, 1137.5304429212786, 1148.1739151223037, 1158.1413279604674, 1168.4068665830748]
ts_best_score = [-3487.1039603960394, -2486.707920792079, -1959.1211024087488, -1458.9972168858678, -1206.6088415034787, -973.5098316024885, -917.905801818839, -864.7647048956395, -788.2437432095469, -739.1836534546751, -694.6351826456148, -645.1751501888199, -575.063836882467, -555.4482202488875, -512.46910385481, -492.98188174169957, -448.20295806326214, -405.55348110535004, -380.4862490428529, -338.7450618530623, -310.12821014959866, -292.2191892211505, -275.9340898399071, -226.1985581428045, -184.74643710773276, -168.13544872530505, -118.6815541388008, -77.93203458742187, -38.594108749994795, 5.249718228811389, 38.69203737360011, 63.50970137257482, 98.12084562652495, 113.58770681163703, 154.24622830855674, 188.72496553327522, 198.5969215097728, 226.9983293834684, 267.629690389614, 308.43258478800124, 330.61274247105837, 367.7649573962524, 395.9610980744723, 430.20046596172307, 439.64945350318993, 451.6043899056925, 479.9072032043866, 498.07356473627686, 532.5882775625134, 547.9174461624209, 563.2153068605649, 597.6192898446704, 605.5681667441477, 623.7067282814049, 635.8825706077705, 670.2562909919395, 672.2246292127525, 681.3286441368139, 690.5163141000227, 699.7834358216588, 734.02059782216, 739.3002551956007, 739.3002551956007, 763.4457808711098, 787.5528378525814, 796.7855210046087, 824.9795786849822, 833.0761273641868, 855.1461699004373, 855.1461699004373, 869.1995361501374, 887.2870403367822, 896.4540321090908, 896.4540321090908, 898.4470353811924, 922.6140271535012, 950.760201396274, 953.8886779088222, 986.9499292596822, 990.0435187903954, 999.1967797708289, 1011.161275796751, 1051.5445270601624, 1069.640340798407, 1072.7339303291199, 1084.7613621459675, 1096.7655492237375, 1108.8158446947602, 1130.9541534260366, 1149.0724659157895, 1149.0724659157895, 1149.0724659157895, 1192.1018112007562, 1200.153019695631, 1202.0975154885061, 1204.090890121325, 1222.198224509887, 1225.2844205369136, 1225.2844205369136, 1234.4920910994294, 1234.4920910994294, 1243.6583361481132, 1261.7177679444808, 1267.709544642988, 1267.709544642988, 1279.7462647851978, 1279.7462647851978, 1281.7588119626569, 1283.7872672767128, 1283.7872672767128, 1310.784297048352, 1313.8642540641094, 1315.882135256699, 1317.8920937014532, 1329.929906574619, 1356.9232812074379, 1396.0147129833256, 1396.0147129833256, 1396.0147129833256, 1402.9622876390417, 1420.9990077812515, 1423.023675401683, 1426.0746778646135, 1432.1223566277406, 1444.1543566072605, 1444.1543566072605, 1447.2483867617273, 1456.3858741925476, 1456.3858741925476, 1456.3858741925476, 1456.3858741925476, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1474.4422932624002, 1482.46946254528, 1494.490533546327, 1515.4429801902727, 1515.4429801902727, 1558.4941886851475, 1558.4941886851475, 1560.4783321995392, 1562.4207848387691, 1562.4207848387691, 1574.4713084231842]

"""
Initial generation
"""
ga_avg_score_init = [2163.940459651346, 2172.6300341087162, 2174.912923060679, 2175.94812044285, 2178.435442392648, 2179.385575413754, 2181.4616321832063, 2184.576091515396, 2185.2121575754563, 2186.8979530733905, 2189.280523012053, 2189.3243458245147, 2190.2061638567097, 2190.2061638567097, 2193.154387230531, 2193.154387230531, 2194.64141841503, 2196.0712957276214, 2197.7420894452875, 2198.6861166926856, 2199.5422002672676, 2199.9385515049735, 2201.257909829782, 2203.774605389316, 2205.5610404296067, 2207.637608459106, 2207.834695420915, 2208.2998479186017, 2210.1407516055874, 2211.847207064075, 2212.776370441268, 2213.747589828637, 2213.747589828637, 2213.747589828637, 2213.747589828637, 2214.6967395371657, 2215.693770523484, 2215.693770523484, 2216.462710807614, 2216.486820378307, 2217.3898644018764, 2218.6824235709614, 2218.936337636702, 2219.4755106039524, 2219.815590243434, 2220.7478979960206, 2221.185872824294, 2221.289706721077, 2221.880392539576, 2221.880392539576, 2222.1169427067853, 2222.1169427067853, 2222.1169427067853, 2222.1169427067853, 2222.5989281196967, 2222.5989281196967, 2222.8123826523506, 2222.8123826523506, 2222.8123826523506, 2222.857917765046, 2223.250870225468, 2223.250870225468, 2223.353972845701, 2223.7601402734263, 2223.9944332700015, 2223.9944332700015, 2223.9944332700015, 2223.9944332700015, 2224.1263836761796, 2225.1936615836153, 2225.4121145914896, 2225.4121145914896, 2225.4121145914896, 2225.4121145914896, 2225.4121145914896, 2226.06249864807, 2226.811903990615, 2226.811903990615, 2226.811903990615, 2226.811903990615, 2226.811903990615, 2227.247955825914, 2227.247955825914, 2227.3469542259545, 2227.746725817821, 2227.746725817821, 2227.8462549776414, 2228.1341823465095, 2228.1341823465095, 2228.335125125437, 2228.335125125437, 2228.335125125437, 2228.335125125437, 2228.335125125437, 2228.335125125437, 2228.8358682013295, 2228.8358682013295, 2229.035084262149, 2229.133570119017, 2229.3326844908343]
ga_best_score_init = [2197.7011195891887, 2197.7011195891887, 2197.7011195891887, 2197.7011195891887, 2197.7011195891887, 2197.7011195891887, 2197.7011195891887, 2204.749811558439, 2204.749811558439, 2204.749811558439, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2209.985988713353, 2217.9525506194773, 2217.9525506194773, 2217.9525506194773, 2217.9525506194773, 2220.319099964868, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2231.057785408474, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395, 2239.3236043015395]

sa_avg_score_init = [2077.535174309026, 2077.535174309026, 2077.535174309026, 2077.9351743090256, 2078.2018409756924, 2078.3923171661686, 2078.79766681492, 2079.112938763949, 2079.365156323172, 2079.7751503278873, 2080.1851443326027, 2080.595138337318, 2081.0051323420334, 2081.2151263467485, 2081.425120351464, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2081.6351143561797, 2082.2351143561796, 2082.8351143561795, 2083.4351143561794, 2084.0351143561793, 2084.6351143561797, 2085.2351143561796, 2085.8351143561795, 2086.4351143561794, 2087.0351143561793, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.6351143561797, 2087.8351143561795, 2088.0351143561793, 2088.2351143561796, 2088.4351143561794, 2088.6351143561797, 2088.8351143561795, 2089.0351143561793, 2089.2351143561796, 2089.4351143561794, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.6351143561797, 2089.8351143561795, 2090.0351143561793, 2090.2351143561796, 2090.4351143561794, 2090.6351143561797, 2090.8351143561795, 2091.0351143561793]
sa_best_score_init = [2077.535174309026, 2077.535174309026, 2077.535174309026, 2079.535174309026, 2079.535174309026, 2079.535174309026, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2081.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2087.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2089.635114356179, 2091.635114356179, 2091.635114356179, 2091.635114356179, 2091.635114356179, 2091.635114356179, 2091.635114356179, 2091.635114356179]

ts_avg_score_init =  [2144.61345971183, 2136.4614025669866, 2138.86620958566, 2133.330820582978, 2132.8035850029123, 2135.8464784933262, 2136.007514270505, 2137.5713152818525, 2136.031326588762, 2135.037577211942, 2134.630369603169, 2134.360854004182, 2134.1292532549437, 2132.8584520392787, 2132.623452985581, 2132.8897843338805, 2132.5170348744837, 2133.757517703531, 2133.5246356889447, 2133.8877999850265, 2133.3552295286177, 2133.6941798059243, 2133.9596497589587, 2133.2468337671808, 2133.16447397928, 2133.6079842009553, 2134.1964776624577, 2133.5736851214224, 2133.4601425207943, 2133.224655872194, 2132.9695799193687, 2133.0346969980196, 2132.8608214618603, 2132.2688166934186, 2132.069144858648, 2131.909420305599, 2131.652906569968, 2131.358520123287, 2131.0286327862677, 2131.395781337155, 2131.627586255695, 2131.475487890659, 2131.4668778048454, 2131.636856773425, 2131.9512747445187, 2132.209297529548, 2132.456816938721, 2132.634070066357, 2132.5848265121563, 2132.3806513326304, 2132.4722682746333, 2132.5404420687855, 2132.422699856074, 2132.1633394353776, 2132.1086384355267, 2132.267070216737, 2131.9890555611532, 2132.212082613895, 2132.127139125105, 2131.8652366133274, 2131.8532117448335, 2131.905040973249, 2131.6420935770884, 2131.77151200753, 2131.730915684884, 2131.5284917044355, 2131.6402435339855, 2131.7779897482287, 2131.768642095966, 2131.900664322683, 2131.8343677241546, 2131.9070780319453, 2132.0045289168493, 2131.965889202834, 2132.336317549045, 2132.281892349132, 2132.292686807755, 2132.176385887915, 2132.3001361595534, 2132.310160280114, 2132.5398196646615, 2132.5467469985333, 2132.6010018687766, 2132.7593238386285, 2132.7170086760407, 2132.572804552684, 2132.749189351367, 2132.8211207733366, 2132.75724777334, 2132.959340123885, 2132.9391086504575, 2133.0058087745697, 2133.0711772290356, 2133.229225901604, 2133.1024993169513, 2132.9893733108865, 2133.0315976323614, 2133.0730090752368, 2133.213609460423, 2132.9540255894162]
ts_best_score_init = [2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2158.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845, 2160.118443514845]



# plt.plot(ga_avg_score[:100])
# plt.plot(sa_avg_score[:100])
# plt.plot(ts_avg_score[:100])
# plt.title("Average Score")
# plt.ylabel('Score')
# plt.xlabel('Iteration')
# plt.legend(['GA', 'SA', 'TS'], loc='upper left')
# plt.savefig('img/avg_score.png')

# plt.plot(ga_best_score[:100])
# plt.plot(sa_best_score[:100])
# plt.plot(ts_best_score[:100])
# plt.title("Best Score")
# plt.ylabel('Score')
# plt.xlabel('Iteration')
# plt.legend(['GA', 'SA', 'TS'], loc='upper left')
# plt.savefig('img/best_score.png')

# plt.plot(ga_avg_score_init[:100])
# plt.plot(sa_avg_score_init[:100])
# plt.plot(ts_avg_score_init[:100])
# plt.title("Average Score")
# plt.ylabel('Score')
# plt.xlabel('Iteration')
# plt.legend(['GA', 'SA', 'TS'], loc='upper left')
# plt.savefig('img/avg_score_init.png')

plt.plot(ga_avg_score_init[:100])
plt.plot(sa_avg_score_init[:100])
plt.plot(ts_avg_score_init[:100])
plt.title("Best Score")
plt.ylabel('Score')
plt.xlabel('Iteration')
plt.legend(['GA', 'SA', 'TS'], loc='upper left')
plt.savefig('img/best_score_init.png')