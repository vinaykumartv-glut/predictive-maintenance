# predictive-maintenance

Unplanned engine failures are costly for vehicle owners and fleet operators alike, driving up repair bills, causing
operational downtime, and creating safety risks. This project explores whether engine sensor readings — RPM,
oil and fuel pressure, coolant pressure, and operating temperatures — can be used to anticipate when an engine is
heading toward a faulty condition, rather than reacting after the fact.
At this stage of the project, the underlying engine sensor dataset has been registered to a central data store,
thoroughly explored to understand feature behaviour and its relationship with engine condition, split and prepared
for modelling, and used to train an XGBoost classifier (a widely used, highly accurate machine-learning method)
with hyperparameter tuning (systematically testing different model settings to find the best-performing
combination) and full experiment tracking via MLflow (a tool that logs every experiment so results can be
compared and reproduced). The trained model has been version-controlled and registered to a model repository,
laying the groundwork for the deployment and evaluation work planned in the final phase of the project.
