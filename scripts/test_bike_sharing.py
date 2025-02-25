import effector


effector.


from sklearn import tree
reg = tree.DecisionTreeRegressor(max_depth=3)
reg.fit(X_train, Y_train)

print(tree.export_graphviz(reg))

# regional_fe = effector.RegionalPDP(
#     data=X,
#     model=m.predict,
#     nof_instances="all",
# )
# regional_fe.fit(
#     features="all",
#     candidate_conditioning_features="all",
# )

# partitioners = list(regional_fe.partitioners.values())
# print(all(p is partitioners[0] for p in partitioners))
