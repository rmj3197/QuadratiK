.. _wireless_indoor_localization_dataset:

Wireless Indoor Localization Dataset
=====================================

The `wireless` data frame has 2000 rows and 8 columns. The first 7 variables
report the measurements of the Wi-Fi signal strength received from 7 Wi-Fi routers in an
office location in Pittsburgh (USA). The last column indicates the class labels.

Format
------

A data frame containing the following columns:

- `V1`: signal strength from router 1.
- `V2`: signal strength from router 2.
- `V3`: signal strength from router 3.
- `V4`: signal strength from router 4.
- `V5`: signal strength from router 5.
- `V6`: signal strength from router 6.
- `V7`: signal strength from router 7.
- `V8`: group memberships, from 1 to 4.

Details
-------

The Wi-Fi signal strength is measured in dBm, decibel milliwatts, which is expressed
as a negative value ranging from -100 to 0. The labels correspond to 'conference room',
'kitchen', 'indoor sports room', and 'other'. In total, we have 4 groups with 500 observations each.

Source
------

Bhatt,Rajen. (2017). Wireless Indoor Localization. UCI Machine Learning Repository. https://doi.org/10.24432/C51880.

References
----------

Rohra, J.G., Perumal, B., Narayanan, S.J., Thakur, P., Bhatt, R.B. (2017). User Localization in an Indoor Environment Using Fuzzy Hybrid of Particle Swarm Optimization & Gravitational Search Algorithm with Neural Networks. In: Deep, K., et al. Proceedings of Sixth International Conference on Soft Computing for Problem Solving. Advances in Intelligent Systems and Computing, vol 546. Springer, Singapore. https://doi.org/10.1007/978-981-10-3322-3_27


