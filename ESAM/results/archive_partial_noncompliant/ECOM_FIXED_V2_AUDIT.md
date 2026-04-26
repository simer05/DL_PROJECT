# ECOM Fixed V2 Audit

- dataset: ecom-offers-v2
- split overlap train/val/test: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- removed columns: ['id', 'offer', 'offerdate', 'repeattrips']
- features used: ['chain', 'market']
- majority baseline val/test: 0.714394/0.723298
- logistic baseline val/test: 0.714394/0.723298
- audit_passed: true
- include_in_scientific_claims: true