# How to reproduce

## Compile
√ repro % `swiftc -O conv_weight_gradient_overflow.swift -o convgrad`

## Test where problem does NOT exist
√ repro % `./convgrad 3236 9 128 none none   # relative error 1.1e-6`
   graph dY vs CPU dY rel err=0.000e+00
n= 3236 k=9 C=128 pre=none post=none rel err=1.121e-06

## Test where problem DOES exist
√ repro % `./convgrad 4096 9 128 none none   # relative error 0.747`
   graph dY vs CPU dY rel err=0.000e+00
n= 4096 k=9 C=128 pre=none post=none rel err=7.474e-01
